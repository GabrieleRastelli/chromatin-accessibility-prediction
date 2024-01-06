'''
This script is used for generating data for Deopen training.
'''
import numpy as np
from pyfaidx import Fasta
import hickle as hkl
import argparse
from tqdm import tqdm


def seq_to_mat(seq):
    encoding_matrix = {'a':0, 'A':0, 'c':1, 'C':1, 'g':2, 'G':2, 't':3, 'T':3, 'n':4, 'N':4}
    mat = np.zeros((len(seq),5))
    for i in range(len(seq)):
        mat[i,encoding_matrix[str(seq[i])]] = 1
    mat = mat[:,:4]
    return mat


def seq_to_mat_opt(seq):
    encoding_matrix = {'A': 0, 'c': 1, 'C': 1, 'g': 2, 'G': 2, 't': 3, 'T': 3, 'n': 4, 'N': 4}
    seq = np.array(list(str(seq).upper()))  # Convert to uppercase and convert to numpy array
    mat = np.eye(5, dtype=int)[[encoding_matrix.get(base, 4) for base in seq]]
    mat = mat[:, :4]
    return mat


def seq_to_kspec_opt(seq, K=6):
    encoding_matrix = {'a': 0, 'A': 0, 'c': 1, 'C': 1, 'g': 2, 'G': 2, 't': 3, 'T': 3, 'n': 0, 'N': 0}
    seq = np.array([encoding_matrix.get(str(base), 0) for base in seq], dtype=np.int8)
    kspec_vec = np.zeros((4 ** K,), dtype=np.int32)

    index = np.dot(seq[:K], 4 ** np.arange(K - 1, -1, -1, dtype=np.int32))

    kspec_vec[index] += 1

    base_power = 4 ** (K - 1)

    for i in range(K, len(seq)):
        index = (index - seq[i - K] * base_power) * 4 + seq[i]
        kspec_vec[index] += 1

    return kspec_vec.reshape(-1, 1)


# assemble all the features into a dictionary
def get_all_feats(spot, genome, label):
    ret = {}
    ret['spot'] = spot
    ret['seq'] = genome[spot[0]][int(spot[1]):int(spot[2])]
    ret['mat'] = seq_to_mat_opt(ret['seq'])
    ret['kmer'] = seq_to_kspec_opt(ret['seq'])
    ret['y'] = label
    return ret


# save the preprocessed data in hkl format
def save_dataset(origin_dataset, save_dir):
    dataset = {}
    for key in origin_dataset[0].keys():
        dataset[key] = [item[key] for item in origin_dataset]
    dataset['seq'] = [str(item).encode('ascii', 'ignore') for item in dataset['seq']]
    for key in origin_dataset[0].keys():
       dataset[key] = np.array(dataset[key])
    hkl.dump(dataset, save_dir, mode='w', compression='gzip')
    print('Training data generation is finished!')


# generate dataset
def generate_dataset(bed_file, genome_file, target, output_folder, sample_length=2114, save_frequency=10):
    dataset = []
    genome = Fasta(genome_file)
    label = np.load(target)
    dataset_saved = 0
    with open(bed_file, 'r') as f_pos:
        lines = list(f_pos)
        ProgressBar = tqdm(enumerate(lines), total=len(lines), desc="Processing lines")
        for i, line in ProgressBar:
            ProgressBar.set_description("Bed line %d" % i)
            chrom = line.rstrip('\n').split('\t')[0]
            start = int(line.rstrip('\n').split('\t')[1])
            end = int(line.rstrip('\n').split('\t')[2])
            mid = (start + end) / 2
            dataset.append(get_all_feats([chrom, mid - sample_length / 2, mid + sample_length / 2], genome, label[1][i]))

            # Check if the length of the dataset is a multiple of save_frequency
            if len(dataset) % save_frequency == 0:
                save_dataset(dataset, f'{output_folder}/saved_dataset_{dataset_saved}.hkl')
                dataset_saved+=1
                dataset = []  # Reset the dataset after saving

    # Save the remaining data if it's not a multiple of save_frequency
    if len(dataset) > 0:
        save_dataset(dataset, f'{output_folder}/saved_dataset_{dataset_saved}.hkl')
        dataset_saved+=1
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deopen data generation')
    parser.add_argument('-bed', dest='bed', type=str, help='input bed file', default='../data/processed/consensus_peaks_inputs.bed')
    parser.add_argument('-genome', dest='genome', type=str, help='genome file in fasta format', default='../data/raw/genome.fa')
    parser.add_argument('-target', dest='target', type=str, help='targets in npy format', default='../data/processed/targets.npy')
    parser.add_argument('-l', dest='length', type=int, default=2114, help='sequence length')
    parser.add_argument('-out', dest='output', type=str, help='output file', default='../data/processed')
    args = parser.parse_args()
    generate_dataset(args.bed, args.genome, args.target, args.output, args.length)
