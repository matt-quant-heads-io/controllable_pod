import argparse

from tensorflow.keras.callbacks import ModelCheckpoint

import utils
import conditional_cnn_model


def get_args():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--obs_size", type=int, required=True)
    argument_parser.add_argument("--data_size", type=int, required=True)
    argument_parser.add_argument("--num_conditions", default=4, type=int)
    argument_parser.add_argument("--action_dim", default=8, type=int)
    argument_parser.add_argument("--epochs", default=100, type=int)
    argument_parser.add_argument("--steps_per_epoch", default=64, type=int)

    return argument_parser.parse_args()


def main(obs_size, data_size, num_conditions, action_dim, epochs, steps_per_epoch):
    conditional_cnn_model_abs_path = (
        f"models/cond_cnn_model_obs_{obs_size}_goal_size_{data_size}.h5"
    )
    cond_cnn_model = conditional_cnn_model.get_conditional_cnn_model(
        obs_size, action_dim, num_conditions
    )

    data, targets = utils.get_train_data_from_csv(obs_size, data_size)

    mcp_save = ModelCheckpoint(
        conditional_cnn_model_abs_path,
        save_best_only=True,
        monitor="categorical_accuracy",
        mode="max",
    )
    history = cond_cnn_model.fit(
        data,
        targets,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        verbose=2,
        callbacks=[mcp_save],
    )


if __name__ == "__main__":
    args = get_args()
    main(
        args.obs_size,
        args.data_size,
        args.num_conditions,
        args.action_dim,
        args.epochs,
        args.steps_per_epoch,
    )
