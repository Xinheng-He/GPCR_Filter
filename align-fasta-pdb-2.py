import os
import pandas as pd
from Bio import PDB
from Bio.Align import PairwiseAligner
from Bio.PDB.Polypeptide import protein_letters_3to1
import argparse

# read pdb
def extract_atom_sequence(pdb_file):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    seq_dict = {}
    for model in structure:
        for chain in model:
            seq = []
            for res in chain:
                if PDB.is_aa(res, standard=True):
                    seq.append(protein_letters_3to1[res.get_resname()])
            seq_dict[chain.id] = "".join(seq)
    return seq_dict


# align pdb & fasta
def align_sequences(uniprot_seq, pdb_seq):
    aligner = PairwiseAligner()
    aligner.mode = "global"
    alignment = aligner.align(uniprot_seq, pdb_seq)[0]
    aligned_uniprot = alignment[0]
    aligned_pdb = alignment[1]
    print("UniProt 对齐后序列:", aligned_uniprot)
    print("PDB 对齐后序列:", aligned_pdb)
    return aligned_uniprot, aligned_pdb


# map
def map_uniprot_to_pdb(aligned_uniprot, aligned_pdb, uniprot_index):
    # Step 1: UniProt origin index -> aligned index
    count, aligned_index = 0, -1
    for i, aa in enumerate(aligned_uniprot):
        if aa != "-":
            count += 1
        if count == uniprot_index:
            aligned_index = i
            break
    if aligned_index == -1 or aligned_pdb[aligned_index] == "-":
        return None

    # Step 2: aligned index -> PDB origin index
    count_pdb, pdb_residue_index = 0, -1
    for i, aa in enumerate(aligned_pdb):
        if aa != "-":
            count_pdb += 1
        if i == aligned_index:
            pdb_residue_index = count_pdb
            break

    return pdb_residue_index

# read highlight indexes from prediction file
def read_highlight_positions(file_path):
    positions = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(":")
            position = int(parts[0].strip().split()[1])
            positions.append(position)
    return positions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protein_id_uniprot", type=str)
    parser.add_argument("--protein_id_pdb", type=str)
    parser.add_argument("--batch_id", type=str)
    parser.add_argument("--predict_dir", type=str)
    args = parser.parse_args()

    # read UniProt seq
    id_target = pd.read_csv('data/idmapping_target.csv')
    id_target.set_index('Target UniProt ID', inplace=True)
    uniprot_seq = id_target.loc[args.protein_id_uniprot, 'Target Sequence']
    print("UniProt 序列:", uniprot_seq)

    # read PDB seq
    pdb_file = os.path.join('pdbs', f'{args.protein_id_pdb}.pdb')
    pdb_sequences = extract_atom_sequence(pdb_file)
    pdb_chain = list(pdb_sequences.keys())[0]
    pdb_seq = pdb_sequences[pdb_chain]
    print("PDB 提取的序列:", pdb_seq)

    # highlight_file = os.path.join(args.predict_dir, 'filtered_test_visual', f'{args.batch_id}-{args.protein_id_uniprot}-{args.protein_id_pdb.upper()}-CA.txt')
    # highlight_positions = read_highlight_positions(highlight_file)

    # res = []
    # aligned_uniprot, aligned_pdb = align_sequences(uniprot_seq, pdb_seq)
    # for uniprot_index in highlight_positions:
    #     pdb_residue_index = map_uniprot_to_pdb(aligned_uniprot, aligned_pdb, uniprot_index)
    #     if pdb_residue_index is not None:
    #         res.append(f'{str(pdb_residue_index)}: {pdb_seq[pdb_residue_index]}')

    # res = ('\n').join(res)
    # os.makedirs(os.path.join(args.predict_dir, 'align'), exist_ok=True)
    # with open(os.path.join(args.predict_dir, 'align', f'{str(args.batch_id)}-{args.protein_id_uniprot}-{args.protein_id_pdb}-CA.txt'), 'w') as f:
    #     f.write(res)

    print(pdb_seq[149:160])