from openpi.training import config
from openpi.policies import policy_config
from openpi.training import data_loader
from openpi.shared import download

if __name__ == '__main__':
    config = config.get_config("pi0_libero")
    checkpoint_dir = "/home/zhiyu/mzh/pi0_base"

    policy = policy_config.create_trained_policy(config, checkpoint_dir)

    data_loader = data_loader.create_data_loader(
        config,
        shuffle=True,
    )

    data_iter = iter(data_loader)

    observation, action = next(data_iter)

    action_chunk = policy.infer(observation)