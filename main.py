import numpy as np
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d
import os
import logging
import pdb

import warnings
warnings.filterwarnings("ignore")



if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, filename='/home/sid/pillarnext/Density_opt/log.log')
    logger = logging.getLogger()

    theta_init = [1,1,1,1,1,1] # Initial sampling percentage
    prob_init = [0,0,0,0,0,0] # Initial posterior probabilities

    num_iter = 500

    # Model configs
    cfg_file = "/home/sid/pillarnext/Open3D-ML/ml3d/configs/pointpillars_kitti.yml"
    cfg = _ml3d.utils.Config.load_from_file(cfg_file)
    ckpt_folder = "/home/sid/pillarnext/Density_opt"
    ckpt_path = os.path.join(ckpt_folder,"pointpillars_kitti_202012221652utc.pth")

    cfg.dataset['dataset_path'] = "/home/sid/kitti_detection_small"
    theta_opt = theta_init.copy()

    # Find out best mAP without resampling
    dataset = ml3d.datasets.KITTI(cfg.dataset.pop('dataset_path', None), **cfg.dataset)
    model = ml3d.models.PointPillars(**cfg.model)
    pipeline = ml3d.pipelines.ObjectDetection(model, dataset=dataset, device="gpu", **cfg.pipeline)
    pipeline.load_ckpt(ckpt_path=ckpt_path)

    logger.info("mAP of the pretrained PointPillars model on the validation set:")
    best_mAP = pipeline.run_valid()
    logger.info("Overall mAP = {}".format(best_mAP))
    logger.info("-"*80)
    # np.random.seed(20)
    best_mAP = 0
    mAP_prev = 0
    theta_prev = theta_init.copy()
    logger.info("MCMC Iterations begin")

    for i in range(num_iter):
        theta = theta_prev.copy()
        delta = 0.05
        sign = 1 if np.random.binomial(1,0.5) == 1 else -1
        delta = sign * delta

        # Proposal theta
        idx = np.clip(np.absolute(np.round(np.random.normal(0,0.5))),0,4)
        theta[int(idx)] = np.clip(theta[int(idx)] + delta,0,1)

        # modified dataset with proposal theta
        cfg.dataset['dataset_path'] = "/home/sid/kitti_detection_small"
        dataset = ml3d.datasets.KITTI(cfg.dataset.pop('dataset_path', None), **cfg.dataset, theta = theta)

        # Initialize detector f with pretrained weights
        model = ml3d.models.PointPillars(**cfg.model)
        pipeline = ml3d.pipelines.ObjectDetection(model, dataset=dataset, device="gpu", **cfg.pipeline)
        pipeline.load_ckpt(ckpt_path=ckpt_path)

        # Finetune for 1 epoch and then evaluate
        mAP_prop = pipeline.run_train(ckpt_path=ckpt_path)
        logger.info("Iteration number {}".format(i+1))
        logger.info("mAP: {}".format(mAP_prop))
        logger.info("theta (Sampling percentage): {}".format(theta))
        if mAP_prop>best_mAP:
            logger.info("Theta accepted (Better theta found)")
            best_mAP = mAP_prop
            theta_opt = theta
            mAP_prev = mAP_prop
            theta_prev = theta
        else:
            u = np.random.uniform(0.0,1.0)
            if min(1.0,mAP_prop/mAP_prev)>u:
                logger.info("Theta accepted")
                mAP_prev = mAP_prop
                theta_prev = theta
            else:
                logger.info("Theta rejected")

        logging.info("Best mAP:{}, Optimal theta:{}".format(best_mAP, theta_opt))













