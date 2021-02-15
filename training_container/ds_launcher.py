import sys
import os
import json
import subprocess
import logging
import argparse

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def _is_master_host() -> bool:

    if os.environ.get('SM_HOSTS', None) is not None:
        hosts =  json.loads(os.environ['SM_HOSTS'])
    else:
        hosts = ["localhost"]    
    
    if len(hosts)==1:
        # to handle single node training
        logger.debug("single node, master=true")
        return True
    
    is_master = ('algo-1' in os.environ['SM_CURRENT_HOST'])
    logger.debug(f"Is master node={is_master}")
    
    return is_master

def _construct_host_file() -> str:
    """constructs MPI-compatible hostfile with all nodes 
     and number of GPU devices"""
    hostfile=os.path.join(os.environ['SM_INPUT_CONFIG_DIR'],'hostfile')
    if os.environ.get('SM_HOSTS', None) is not None:
        hosts =  json.loads(os.environ['SM_HOSTS'])
    else:
        hosts = ["localhost"]
    
    if len(hosts)==1:
        # for single node training
        # deepspeed will automatically use all local resources
        # no need to create host files
        logger.debug("Single node training, skipping hostfile creation.")
        return ""
    
    with open(hostfile,'a') as f:
        for h in hosts:
            f.write(f"{h} slots={os.environ['SM_NUM_GPUS']}")
    logger.debug(f"Hostfile {hostfile} has been saved.")
    return hostfile

# TODO: remove as it's no longer needed
# All configs are stored in training_container dir
def _construct_config_file() -> str:
    config = json.loads(os.environ['SM_HPS'])
    print(f"Deserialized from HP config: {config}")
    config_file = os.path.join(os.environ['SM_INPUT_CONFIG_DIR'], 'ds_config.json')
    with open(config_file,'w') as f:
        json.dump(config, f)
    logger.debug(f"Config file {config_file} saved.")
    return config_file
    

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--deepspeed-config-file',
                        type=str,
                        help='Deepspeed config file'
    )
    parser.add_argument('--train-script',
                       type=str,
                       default='train_cifar10.py',
                       help='Training script to be executed by DeepSpeed')
    args, unknown = parser.parse_known_args()    
    
    
    if not _is_master_host():
        # worker nodes do nothing as deepspeed launcher
        # will distribute the tasks
        logger.info("Non master node. Stopping the execution.")
        return

    # constructing script launcher based on
    # https://www.deepspeed.ai/getting-started/#launching-deepspeed-training
    train_script = os.path.join(os.environ['SM_MODULE_DIR'], args.train_script)
    hostfile = _construct_host_file()
    deepspeed_config = os.path.join(os.environ['SM_MODULE_DIR'], args.deepspeed_config_file)
    
    cmd = ["deepspeed"]
    if hostfile:
        cmd.append(f"--hostfile {hostfile}")
    cmd.append(train_script)
    cmd.append("--launcher=openmpi") # defaulting to openmpi launcher https://github.com/microsoft/DeepSpeed/blob/c5e4264186c321321858fb46d75c328fdc908ddb/deepspeed/launcher/runner.py#L95
    cmd.append("--deepspeed-flag")
    cmd.append(f"--config-file={deepspeed_config}")
    logger.info(f"Following command line will be launched on master:{cmd}")
    
    # Placeholder. If we need to customize our env, we can
    # update current_env list
    current_env = os.environ.copy()
    logger.info(f"Env variables to be distributed across nodes: {current_env}")

    process = subprocess.Popen(cmd, env=current_env, stderr=subprocess.STDOUT, stdout=subprocess.PIPE)
    
    while True:
        output = process.stdout.readline()

        if process.poll() is not None:                
            break
        if output:
            logger.info(output.decode("utf-8").strip())
    rc = process.poll()    

    if process.returncode != 0:
        raise subprocess.CalledProcessError(returncode=process.returncode,
                                            cmd=cmd)

        

if __name__ == "__main__":
    main()