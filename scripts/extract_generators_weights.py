import argparse
import torch


def extract_generator(state_dict, prefix):
    return {
        k.replace(prefix + ".", ""): v
        for k, v in state_dict.items()
        if k.startswith(prefix)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)

    args = parser.parse_args()

    checkpoint = torch.load(
        args.checkpoint_path,
        weights_only=False,
        map_location="cpu"
    )

    model_state_dict = checkpoint["model_state_dict"]

    generators_state_dict = {
        "generator_A": extract_generator(model_state_dict, "generator_A"),
        "generator_B": extract_generator(model_state_dict, "generator_B"),
    }

    torch.save(generators_state_dict, args.output_path)


if __name__ == "__main__":
    main()
