import ntpath
from glob import glob
import argparse
import os


def count_lines(filesPath):
    count = 0
    for f in glob(filesPath + '/*.txt'):
        count += len(open(f).readlines())
    
    return count
def save_counts(path):
    detectionFolders = glob(path + '/*_*-*_*')
    vidName = [i for i in path.split(os.sep) if 'GOPR' in i][0]
    # print(detectionFolders)
    # print(vidName)
    with open(os.sep.join([path,vidName + '-detectionCounts.csv']), 'w+') as f:
        f.write('segment,count\n')
        for d in detectionFolders:
            segment = ntpath.basename(d)
            count = count_lines(d)
            f.write(','.join([segment, str(count)]) + '\n')
            # print(segment,count)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Count number of lines in text file')
    parser.add_argument('--path', help='Path to txt files')

    args = parser.parse_args()
    save_counts(args.path)        