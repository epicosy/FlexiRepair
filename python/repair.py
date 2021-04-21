import shutil
import os
import logging

from dataclasses import dataclass
from pathlib import Path
from typing import List, AnyStr
from common.commons import shellGitCheckout, get_prioritization, load_zipped_pickle, parallelRunMerge

ROOT_PATH = Path(os.environ["ROOT_DIR"])
DATA_PATH = Path(os.environ["DATA_PATH"])
SPINFER_PATH = os.environ["spinfer"]
DATASET = Path(os.environ["dataset"])
ALL_DATASET = ROOT_PATH / 'data' / 'allCocciPatterns.pickle'
COCCI_PATH = Path(os.environ["coccinelle"]) / 'spatch'
PRIORITIZATION = get_prioritization()


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

        output, e = shellGitCheckout(self.compile_script)
        print(output)
        if e or not output:
            print(f"\nCompilation failed for patch {patch.name}")
            return False

        print(f"\nPatch {patch.name} passed compilation")

        for test_name, outcome, e in iter(self.__iter__()):
            if e or not outcome:
                print(f"\nTest {test_name} failed with return code {outcome} " + (str(e) if e else ''))
            else:
                print(f"\nTest {test_name} passed")

        if self.failed > 0:
            print(f"\nPatch {patch.name} failed {self.failed} tests.")

        return self.is_valid()

    def __iter__(self):
        self.cur = 0
        return self

    def __next__(self):
        if self.cur + 1 == self.total:
            raise StopIteration

        test_name = self.tests[self.cur]
        output, e = shellGitCheckout(self.script.replace("TEST_NAME", test_name))
        print(output)
        if e or not output:
            self.outcomes[test_name] = 1
            self.passed += 1
        else:
            self.outcomes[test_name] = 0
            self.failed += 1

        self.cur += 1

        return test_name, output, e

    def is_valid(self):
        if self.passed == self.total and self.total != 0:
            return True

        return False


@dataclass
class CocciExecutor:
    target: Path
    sp_file: Path
    pattern: str
    patch: Path
    spatch_file: Path

    def __call__(self, *args, **kwargs):
        if self.run_spatch():
            if self.run_gnu_patch():
                return self.patch
        return None

    def run_spatch(self):
        # check if cocci file exists, otherwise create tmp file with the pattern
        if not self.sp_file.exists():
            with self.sp_file.open(mode='w') as cf:
                cf.write(self.pattern)

        spatch_cmd = f"{COCCI_PATH} --sp-file {self.sp_file} {self.target} --patch {self.spatch_file}"
        # spatch_cmd += f" > {self.spatch_file}"
        output, e = shellGitCheckout(spatch_cmd)

        if e is not None:
            logging.warning(e)
            return None

        if not self.spatch_file.exists() or self.spatch_file.stat().st_size == 0:
            self.spatch_file.unlink()
            return None

        return output

    def run_gnu_patch(self):
        output, e = shellGitCheckout(f"patch {self.target} {self.spatch_file} -o {self.patch}")

        if e is not None:
            logging.warning(e)
            return None

        return output


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
        self.load_all()

        cmd_list = [CocciExecutor(target=self.src_file, pattern=pattern,
                                  sp_file=self.data_path / Path(file),
                                  spatch_file=self.patterns_dir / (self.src_file.stem + file.split('.')[0] + '.txt'),
                                  patch=self.patches_dir / (self.src_file.stem + file.split('.')[0] + '.c')) for file, pattern
                    in self.patterns.items()]

        patches = parallelRunMerge(cmd_list)
        self.patches = list(filter(None, patches))

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

    def apply(self, patch: Path):
        print(f"Applying patch {patch.name}")

        with self.source_file.open(mode="w") as sf, patch.open(mode="r") as p:
            sf.write(p.read())


class ProgramRepair:
    def __init__(self, generator: PatchGenerator, validator: Validator):
        self.generator = generator
        self.validator = validator
        self.repair_dir = generator.working_dir / 'repair'

    def __call__(self, *args, **kwargs):
        self.generator.backup()

        for patch in self.generator():
            if patch is None:
                continue
            self.generator.apply(patch)

            if self.validator(patch):
                self.repair(patch)
                print(f"\nProgram repaired with patch {patch}")
                break

        self.generator.restore()

    def repair(self, patch: Path):
        self.repair_dir.mkdir()
        repair_file = self.repair_dir / patch.name

        with repair_file.open(mode="w") as rf, patch.open(mode='r') as p:
            rf.write(p.read())
