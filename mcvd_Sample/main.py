from configs.configration import parse_args_and_config
from runner import NCSNRunner


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    args, config, config_uncond = parse_args_and_config()
    # os.makedirs(os.path.join(args.exp, 'image_samples'), exist_ok=True)
    # args.image_folder = os.path.join(args.exp, 'image_samples', args.image_folder)

    # print(args.image_folder)
    version = getattr(config.model, 'version', "SMLD")
    Runner = NCSNRunner(args, config, config_uncond)