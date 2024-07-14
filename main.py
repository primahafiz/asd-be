from argparse import ArgumentParser
from feature import *
from constant import *
from training import *
from api import startAPI

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--type', type=str,help='Type of process',default='',choices=['extract','train','api'])
    parser.add_argument('--model',type=str,help='Model for training',default='',choices=['autoencoder','unet','idnn','unetidnn'])
    parser.add_argument('--feature',type=str,help='Feature used for extract and model',default='',choices=['logmel','gammatone'])
    parser.add_argument('--path_to_extract',type=str,help='Only for extract',default='')

    args = parser.parse_args()

    if args.type == 'extract':
        resultExtract = None
        if args.feature == FeatureType.LOGMEL.value:
            resultExtract = extractLogMelFromPath(args.path_to_extract)
        elif args.feature == FeatureType.GAMMATONE.value:
            resultExtract = extractGammatoneFromPath(args.path_to_extract)

        print(resultExtract.shape)
        print(resultExtract)
    
    elif args.type == 'train':
        train(args.model,args.feature)

    elif args.type == 'api':
        startAPI()