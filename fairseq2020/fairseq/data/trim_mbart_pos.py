import torch
import os

def run(args):
    SIZE=args.len
    model = torch.load(os.path.join(args.home, "{}/model.pt".format(args.pre_train_dir)))

    print(model["model"]["encoder.embed_positions.weight"].size())
    print(model["model"]["decoder.embed_positions.weight"].size())

    model["model"]["encoder.embed_positions.weight"] = model["model"]["encoder.embed_positions.weight"][:SIZE]
    model["model"]["decoder.embed_positions.weight"] = model["model"]["decoder.embed_positions.weight"][:SIZE]
    torch.save(model, os.path.join(args.home, "{}/model_{}.pt".format(args.pre_train_dir, SIZE-2)))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
       description=__doc__,
       formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--home", type=str, default='', required=True,
                        help="Home where all XWikis related files are.")
    parser.add_argument("--pre-train-dir", type=str, required=True, help="The pre-trained mBART model directory.")
    parser.add_argument('--len', default=600, type=int, metavar='N', help='beam size')

    args = parser.parse_args(sys.argv[1:])

    run(args)