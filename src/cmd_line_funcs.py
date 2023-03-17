import argparse
import torch

if torch.cuda.is_available():
    DEFAULT_DEVICE = 'cuda'
else:
    DEFAULT_DEVICE = 'cpu'


def train_parser():
    argP = argparse.ArgumentParser()
    argP.add_argument("-d", "--device", type=torch.device,
                      default=torch.device(DEFAULT_DEVICE),
                      help="Maximum number of training epochs to run (default: %s)." % DEFAULT_DEVICE)
    argP.add_argument("-e", "--maxepoch", type=int, default=100,
                      help="Maximum number of training epochs to run (default: %(default)s).")
    argP.add_argument("-b", "--batch-size", type=int, default=100,
                      help="Batch size (default: %(default)s).")
    argP.add_argument("-w", "--early-stop-window", type=int, default=4,
                      help="rolling mean window for early stopping (default: %(default)s).")
    argP.add_argument("-s", "--binary-start", type=int, default=0,
                      help="Pretrain a binary model to use transfer learning, running this many epochs.")
    argP.add_argument("-n", "--model-number", type=int, default=1,
                      help="The number for the model, only affecting the output name of the saved model (default: %(default)s).")
    argP.add_argument("--pretrained", action="store_true",
                      help="Use a pretrained BEiT model.")
    argP.add_argument("--train-all", action="store_true",
                      help="Train also the pretrained BEiT model.")
    argP.add_argument("-lr", "--learning-rate", type=float, default=0.0005,
                      help="Set the learning rate (default: %(default)s).")
    argP.add_argument("-wd", "--weight-decay", type=float, default=0.001,
                      help="Set the weight decay (l2-regularisation) (default: %(default)s).")
    argP.add_argument("--use-best-params", action="store_true",
                      help="Take the model parameters from the epoch with the best dev metric.")
    argP.add_argument("--slurm-mode", action="store_true",
                      help="Run in SLURm mode (changes stdout output only).")
    return argP.parse_args()


def test_parser():
    argP = argparse.ArgumentParser()
    argP.add_argument("-d", "--device", type=torch.device,
                      default=torch.device(DEFAULT_DEVICE),
                      help="Maximum number of training epochs to run (default: %s)." % DEFAULT_DEVICE)
    argP.add_argument("-b", "--batch-size", type=int, default=100,
                      help="Batch size (default: %(default)s).")
    argP.add_argument("-n", "--model-number", nargs="+", type=int, default=[1],
                      help="The number for the model, afecting which model file is loaded (default: %(default)s).")
    argP.add_argument("--pretrained", action="store_true",
                      help="Load a pretrained BEiT model.")
    return argP.parse_args()
