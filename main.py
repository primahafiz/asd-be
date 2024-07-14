from argparse import ArgumentParser
from feature import *
from constant import *

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--type', type=str,help='Type of process',default='')
    parser.add_argument('--model',type=str,help='Model for training',default='')
    parser.add_argument('--feature',type=str,help='Feature used for extract and model',default='')
    parser.add_argument('--path_to_extract',type=str,default='')

    args = parser.parse_args()

    if args.type == 'extract':
        resultExtract = None
        if args.feature == FeatureType.LOGMEL.value:
            resultExtract = extractLogMelFromPath(args.path_to_extract)
        elif args.feature == FeatureType.GAMMATONE.value:
            resultExtract = extractGammatoneFromPath(args.path_to_extract)
            
        print(resultExtract.shape)
        print(resultExtract)
    