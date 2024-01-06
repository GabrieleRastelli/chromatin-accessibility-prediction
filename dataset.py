from torch.utils.data import DataLoader
from public.loader import Seq2Ab, SampleLoader
import os.path
import numpy as np


class MyDataset(Seq2Ab):
    def __init__(self, mode):
        super(MyDataset, self).__init__()

        self.all_regions = self._load_bed_file('data/processed/consensus_peaks_inputs.bed')

        self.split_dict = {"val": ["chr8", "chr10"], "infer": ["chr9", "chr18"]}

        indices = self._get_indices_for_set_type(
            self.split_dict, mode, self.all_regions, 1.0
        )
        super().load_data_hkl(sorted([os.path.join('data/processed', f) for f in os.listdir('data/processed') if f.endswith('.hkl')]))
        super().set_input_and_target(indices)

    def __getitem__(self, item):
        cur_x = self.x[item]
        cur_label = self.labels[item]
        return cur_x, cur_label

    def __len__(self):
        return len(self.x)

    def _get_indices_for_set_type(
        self, split_dict: dict, set_type: str, all_regions: list, fraction_of_data=1.0
    ):
        """Get indices for the specified set type (train/val/infer)."""
        if set_type not in ["learn", "val", "infer"]:
            raise ValueError("set_type must be 'train', 'val', or 'infer'")

        if set_type == "learn":
            excluded_chromosomes = set(
                split_dict.get("val", []) + split_dict.get("infer", [])
            )
            all_chromosomes = set(chrom for chrom, _, _ in all_regions)
            selected_chromosomes = all_chromosomes - excluded_chromosomes
        else:
            selected_chromosomes = set(split_dict.get(set_type, []))

        indices = [
            i
            for i, region in enumerate(all_regions)
            if region[0] in selected_chromosomes
        ]
        if fraction_of_data != 1.0:
            indices = indices[: int(np.ceil(len(indices) * fraction_of_data))]
        return indices

    def _load_bed_file(self, regions_bed_filename: str):
        """
        Read BED file and yield a region (chrom, start, end) for each invocation.
        """
        regions = []
        with open(regions_bed_filename, "r") as fh_bed:
            for line in fh_bed:
                line = line.rstrip("\r\n")

                if line.startswith("#"):
                    continue

                columns = line.split("\t")
                chrom = columns[0]
                start, end = [int(x) for x in columns[1:3]]
                regions.append((chrom, start, end))
        return regions


class MyDataLoader(SampleLoader):
    def __init__(self, mode="learn", batch_size=None):
        super(MyDataLoader, self).__init__()

        shuffle = True if mode == "learn" else False
        drop_last = True if mode == 'learn' else False

        self.loader = DataLoader(dataset=MyDataset(mode),
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 num_workers=0,
                                 drop_last=drop_last)

    def __call__(self, *args, **kwargs):
        return self.loader
