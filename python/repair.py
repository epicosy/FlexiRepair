import shutil
import os
import numpy as np

from pathlib import Path
from typing import List, AnyStr
from common.commons import shellGitCheckout, get_prioritization, load_zipped_pickle

DATA_PATH = Path(os.environ["DATA_PATH"])
SPINFER_PATH = os.environ["spinfer"]
DATASET = Path(os.environ["dataset"])
COCCI_PATH = Path(os.environ["coccinelle"]) / 'spatch'
# TODO: FIX this prioritization not working
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

        if output != 0 or e is not None:
            print(f"\nCompilation failed for patch {patch.name}")
            return False

        print(f"\nPatch {patch.name} passed compilation")

        for test_name, outcome, e in iter(self.__iter__()):
            if outcome != 0 or e is not None:
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
        self.outcomes[test_name] = int(output)

        if int(output) == 0:
            self.passed += 1
        else:
            self.failed += 1

        self.cur += 1

        return test_name, output, e

    def is_valid(self):
        if self.passed == self.total and self.total != 0:
            return True

        return False


class CocciPatches:
    def __init__(self, src_file: Path, patterns_dir: Path, patches_dir: Path):
        self.src_file = src_file
        self.patterns_dir = patterns_dir
        self.patches_dir = patches_dir
        self.patterns = []
        self.patches = []

    def __call__(self):
        if self.patches:
            return self.patches

        self.init_dirs()
        self.load()

        return iter(self.__iter__())

    def __iter__(self):
        self.cur = 0
        return self

    def __next__(self):
        if self.cur + 1 == len(self.patterns):
            raise StopIteration
        patch = None

        while patch is None:
            patch = self.create_patch()

            if self.cur + 1 == len(self.patterns):
                break

            self.cur += 1

        return patch

    def create_patch(self):
        patch_name = self.patterns_dir / self.src_file.name
        pattern = self.patterns[self.cur]
        # TODO: FIX the parallel problem
        patch_file = self.patterns_dir / (self.src_file.stem + pattern.stem + '.txt')
        patched = self.patches_dir / (self.src_file.stem + pattern.stem + '.c')
        sp_file = DATASET / 'cocci' / pattern.name

        cocci_cmd = f"{COCCI_PATH} --sp-file {sp_file} {self.src_file} --patch -o {patch_name} > {patch_file}"
        patch_cmd = f"patch -d {self.src_file} -i {patch_file} -o {patched}"
        output, e = shellGitCheckout(cocci_cmd)

        if e is not e:
            return None

        if patch_file.stat().st_size == 0:
            patch_file.unlink()
            return None

        output, e = shellGitCheckout(patch_cmd)

        if e is not None:
            return None
        self.patches.append(patched)
        return patched

    def load(self):
        sp_files = load_zipped_pickle(DATA_PATH / 'uPatterns.pickle')
        sp_files.sort_values(by=PRIORITIZATION, inplace=True, ascending=False)
        sp_files = sp_files.loc[sp_files['uid'] != '.DS_Store']

        self.patterns = sp_files[['uid']].values.tolist()

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
