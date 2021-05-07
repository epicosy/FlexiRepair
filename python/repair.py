import shutil
import os
import logging

from dataclasses import dataclass
from pathlib import Path
from typing import List, AnyStr
from common.commons import shellGitCheckout, get_prioritization, load_zipped_pickle, parallelRunMerge
from common.preprocessing import getTokensForPatterns

ROOT_PATH = Path(os.environ["ROOT_DIR"])
DATA_PATH = Path(os.environ["DATA_PATH"])
SPINFER_PATH = os.environ["spinfer"]
DATASET = Path(os.environ["dataset"])
ALL_DATASET = ROOT_PATH / 'data' / 'allCocciPatterns.pickle'
COCCI_PATH = Path(os.environ["coccinelle"]) / 'spatch'
PRIORITIZATION = get_prioritization()

c_code_keywords = {'auto', 'else', 'long', 'switch', 'break', 'enum', 'register', 'typedef', 'case', 'extern', 'return',
                   'union', 'char', 'float', 'short', 'unsigned', 'const', 'for', 'signed', 'void', 'continue', 'goto',
                   'sizeof', 'volatile', 'default', 'if', 'static', 'while', 'do', 'int', 'struct', 'double'}


def count_keywords(tokens_set: set):
    """
        count the number of c code keywords in a list of tokens

        Parameters
        ----------
        tokens_set: Set[AnyStr]
            The file location of the spreadsheet

        Returns
        -------
        count: int
            the count of strings used that are the header columns
    """
    return len(tokens_set.intersection(c_code_keywords))


class Validator:
    def __init__(self, test_script: str, compile_script: str, tests: List[AnyStr]):
        self.total = len(tests)
        self.tests = tests
        self.script = test_script
        self.compile_script = compile_script

    def __call__(self, patch: Path):
        self.passed = 0
        self.failed = 0
        self.outcomes = {t: -1 for t in self.tests}

        print(f"Compiling patch: {patch}")
        output, e = shellGitCheckout(self.compile_script)
        print(f"Compilation output: {output}")

        if e or not output:
            logging.warning(f"\nCompilation failed for patch {patch.name}")
            return False

        print(f"\nPatch {patch.name} compiled")

        _ = [r for r in iter(self.__iter__())]

        if self.failed > 0:
            print(f"\nPatch {patch.name} failed {self.failed} tests.")

        return self.is_valid()

    def __iter__(self):
        self.cur = 0
        return self

    def __next__(self):
        if self.cur == self.total:
            raise StopIteration

        test_name = self.tests[self.cur]
        print(f"Testing {test_name}")
        output, e = shellGitCheckout(self.script.replace("TEST_NAME", test_name))

        if e or not output:
            logging.warning(f"\nTest {test_name} failed with return {e}")
            self.outcomes[test_name] = 0
            self.failed += 1
        else:
            print(f"Test {test_name} passed.")
            self.outcomes[test_name] = 1
            self.passed += 1

        self.cur += 1

        return test_name, output, e

    def is_valid(self):
        if self.passed == self.total and self.total != 0:
            return True

        return False

    def success_rate(self):
        if self.total != 0:
            return round(self.passed / self.total, 3)
        return 0


@dataclass
class Patch:
    file: Path
    size: int
    similarity: float
    keywords: int


@dataclass
class CocciExecutor:
    target: Path
    sp_file: Path
    pattern: str
    patch_file: Path
    spatch_file: Path

    def __call__(self, *args, **kwargs):
        # check if cocci file exists, otherwise create tmp file with the pattern
        if not self.sp_file.exists():
            with self.sp_file.open(mode='w') as cf:
                cf.write(self.pattern)
        # if pattern matches with code in the target file, then the function returns the size in tokens of the pattern,
        # the ratio of matching tokens, and the number of matching keywords
        pattern_size, similarity, keywords = self.target_has_pattern()

        if similarity:
            spatch_cmd = f"{COCCI_PATH} --sp-file {self.sp_file} {self.target} > {self.spatch_file}"
            output, e = shellGitCheckout(spatch_cmd)

            if not self.spatch_file.exists():
                return None

            if self.spatch_file.stat().st_size == 0:
                self.spatch_file.unlink()
                return None

            output, e = shellGitCheckout(f"patch {self.target} {self.spatch_file} -o {self.patch_file}")

            if not self.patch_file.exists():
                return None

            return Patch(file=self.patch_file, size=pattern_size, similarity=similarity, keywords=keywords)

        return None

    def target_has_pattern(self):
        with self.target.open(mode='r') as t:
            lines = t.read()
        target_tokens = getTokensForPatterns(lines)
        pattern_tokens = getTokensForPatterns(self.pattern)
        intersected_tokens = set(target_tokens).intersection(set(pattern_tokens))

        if len(intersected_tokens) > 0:
            return len(pattern_tokens), len(intersected_tokens) / len(pattern_tokens), count_keywords(pattern_tokens)

        return None, None, None


class CocciPatches:
    def __init__(self, src_file: Path, patterns_dir: Path, patches_dir: Path):
        self.src_file = src_file
        self.patterns_dir = patterns_dir
        self.patches_dir = patches_dir
        self.patterns = {}
        self.patches = []
        self.data_path = None

    def __call__(self):
        if self.patches:
            return self.patches

        self.init_dirs()
        self.load()

        print(f"Generating patches for {self.src_file}")

        cmd_list = [CocciExecutor(target=self.src_file, pattern=pattern,
                                  sp_file=self.data_path / Path(file),
                                  spatch_file=self.patterns_dir / f"{self.src_file.stem}_{file}.txt",
                                  patch_file=self.patches_dir / f"{self.src_file.stem}_{file}.c") for file, pattern
                    in self.patterns.items()]

        patches = parallelRunMerge(cmd_list, max_workers=2)
        self.patches = list(filter(None, patches))
        print(f"Generated {len(self.patches)} patches")
        # sort patches by similarity and size
        self.patches.sort(key=lambda p: p.similarity * p.size * p.keywords)

        return self.patches

    def load(self):
        sp_files = load_zipped_pickle(DATA_PATH / 'uPatterns.pickle')
        sp_files.sort_values(by=PRIORITIZATION, inplace=True, ascending=False)
        sp_files = sp_files.loc[sp_files['uid'] != '.DS_Store']
        self.patterns = {row.uid: row.pattern for row_id, row in sp_files.iterrows()}
        self.data_path = DATASET / 'cocci'

    def load_all(self):
        sp_files = load_zipped_pickle(DATA_PATH / 'allCocciPatterns.pickle')
        self.patterns = {row.cid: row.pattern for row_id, row in sp_files.iterrows()}
        self.data_path = Path('/tmp')

    def init_dirs(self):
        if self.patterns_dir.exists():
            shutil.rmtree(self.patterns_dir)

        self.patterns_dir.mkdir()

        if self.patches_dir.exists():
            shutil.rmtree(self.patches_dir)

        self.patches_dir.mkdir()


class PatchGenerator:
    def __init__(self, working_dir: str, source_file: str):
        self.working_dir = Path(working_dir)
        self.source_file = Path(source_file)
        self.backup_source = self.working_dir / ('backup_' + self.source_file.name)
        self.cocci_patches = CocciPatches(src_file=self.source_file, patterns_dir=self.working_dir / 'patterns',
                                          patches_dir=self.working_dir / 'patches')

    def __call__(self):
        return self.cocci_patches()

    def backup(self):
        with self.backup_source.open(mode="w") as bs, self.source_file.open(mode="r") as sf:
            bs.write(sf.read())

    def restore(self):
        with self.source_file.open(mode="w") as sf, self.backup_source.open(mode="r") as bs:
            sf.write(bs.read())

        self.backup_source.unlink()

    def apply(self, patch_file: Path):
        logging.info(f"Applying patch {patch_file.name}")

        with self.source_file.open(mode="w") as sf, patch_file.open(mode="r") as p:
            sf.write(p.read())


class ProgramRepair:
    def __init__(self, generator: PatchGenerator, validator: Validator):
        self.generator = generator
        self.validator = validator
        self.repair_dir = generator.working_dir / 'repair'
        self.limit = 20

    def __call__(self, *args, **kwargs):
        self.generator.backup()
        success_rate_patches = {}

        for idx, patch in enumerate(self.generator()):
            if patch is None:
                continue
            if idx + 1 == self.limit:
                break
            self.generator.apply(patch.file)
            patch_passes = self.validator(patch.file)
            success_rate_patches[patch.file.name] = (self.validator.success_rate(), self.validator.outcomes)

            if patch_passes:
                self.repair(patch.file)
                logging.info(f"\nProgram repaired with patch {patch.file}")
                break

        if success_rate_patches:
            print("Repair Summary")
            for patch_name, outcomes in success_rate_patches.items():
                print(patch_name, outcomes[0])
                for tn, v in outcomes[1].items():
                    print(f"\t{tn} {v}")

        self.generator.restore()

    def repair(self, patch: Path):
        self.repair_dir.mkdir()
        repair_file = self.repair_dir / self.generator.source_file.name

        with repair_file.open(mode="w") as rf, patch.open(mode='r') as p:
            rf.write(p.read())
