import datetime
import argparse
from daneel.parameters import Parameters
from daneel.detection import *
from daneel.atmosphere import *


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        dest="input_file",
        type=str,
        required=True,
        help="Input par file to pass",
    )

    parser.add_argument(
        "-d",
        "--detect",
        dest="detect",
        type = str,
        required=False,
        help="Initialise detection algorithms for Exoplanets. DETECT =rf/dt/cnn",
    )

    parser.add_argument(
        "-a",
        "--atmosphere",
        dest="atmosphere",
        type = str,
        required=False,
        help="Atmospheric Characterisazion from input transmission spectrum. ATMOSPHERE = model/retrieve",
    )
    
    parser.add_argument(
        "-t",
        "--transit",
        dest="transit",
        required=False,
        help="Plotting the light curve of the transit",
        action="store_true",
    )
    args = parser.parse_args()

    """Launch Daneel"""
    start = datetime.datetime.now()
    print(f"Daneel starts at {start}")

    input_pars = Parameters(args.input_file).params

    if args.detect:
        CSV_PATH = '/home/sijis/Desktop/com_ast/test/Computational-_astro/src/daneel/detection/tess_data.csv'    # change if needed
        N_BINS = 1000
        USE_SCALER = False                  # Random Forest doesn't need scaling; set True if you want comparability
        SAMPLES_PER_CLASS = 350             # per-class size after augmentation (train split only)
        if args.detect =='cnn':
            USE_SCALER = True
        model = ML(CSV_PATH,N_BINS,USE_SCALER,SAMPLES_PER_CLASS,args.model)
        model.main()
    if args.atmosphere:
        if args.atmosphere == 'model':
            atm = Atmosphere(input_pars)
            atm.model()
        if args.atmosphere == 'retrieve':
            atm = Atmosphere(input_pars)
            atm.retrieve()
        	
    if args.transit:
        transit = Transit(input_pars)
        transit.flux() 

    finish = datetime.datetime.now()
    print(f"Daneel finishes at {finish}")


if __name__ == "__main__":
    main()
