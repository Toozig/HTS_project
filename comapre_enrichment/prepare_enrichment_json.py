from .compare_enrichment_utils import *
import json
import re
from collections import defaultdict
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


DEBUG = False

def get_highest_cycle_files(file_list):
    # Regular expression to match the file name pattern
    pattern = re.compile(r'RBP(\d+)_(\d+)\.txt')
    
    # Dictionary to store the highest version for each RBP
    rbp_versions = defaultdict(lambda: (0, ''))
    
    for file_name in file_list:
        match = pattern.match(file_name)
        if match:
            rbp_num = int(match.group(1))
            version = int(match.group(2))
            
            # Update if this version is higher
            if version > rbp_versions[rbp_num][0]:
                rbp_versions[rbp_num] = (version, file_name)
    
    # Return only the file names of the highest versions
    return [file_name for _, file_name in rbp_versions.values()]


def process_sample(args):
    sample_file, samples_dir = args
    sample_name = os.path.splitext(sample_file)[0]
    input_path = os.path.join(samples_dir, sample_file)
    with open(input_path, 'r') as f:
        lines = f.readlines()
        sequences = [line.strip().split(',')[0] for line in lines]
    enrichment = calculate_8mer_enrichment(sequences)
    
    # print(f"Calculated enrichment for {sample_name}")
    return sample_name, enrichment



def main():
    parser = argparse.ArgumentParser(description='Calculate 8-mer enrichment for all samples and save as a single JSON.')
    parser.add_argument('samples_dir', help='Directory containing sample files')
    parser.add_argument('output_file', help='Path to save the output JSON file')
    args = parser.parse_args()

    if not os.path.isdir(args.samples_dir):
        print(f"Error: The directory {args.samples_dir} does not exist.")
        return

    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sample_files = [f for f in os.listdir(args.samples_dir) if f.endswith('.txt')]
    sample_files = get_highest_cycle_files(sample_files)

    all_enrichments = {}


    if not DEBUG:
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = process_map(process_sample, [(f, args.samples_dir) for f in sample_files], max_workers=mp.cpu_count())
            all_enrichments = dict(results)
    else:
        for f in sample_files[:2]:
            sample_name, enrichment = process_sample(f, args.samples_dir)
            all_enrichments[sample_name] = enrichment

    # make sure the output directory exists
    if not os.path.exists(os.path.dirname(args.output_file)):
        os.makedirs(os.path.dirname(args.output_file))
    with open(args.output_file, 'w') as f:
        json.dump(all_enrichments, f)

    print(f"Enrichment calculation complete for all samples. Results saved to {args.output_file}")

if __name__ == "__main__":
    main()