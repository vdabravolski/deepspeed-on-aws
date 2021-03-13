import sys
import os
import json
import subprocess
import logging
import argparse
from contextlib import contextmanager
import signal
import time
import socket

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def common_setup():

    # Read info that SageMaker provides
    current_host = os.environ['SM_CURRENT_HOST']
    hosts = json.loads(os.environ['SM_HOSTS'])

    # Enable SSH connections between containers
    _start_ssh_daemon()

    if not _is_master_host():
        _wait_for_worker_nodes_to_start_sshd(hosts)


def _start_ssh_daemon():
    subprocess.Popen(["/usr/sbin/sshd", "-D"])


def _get_master_host_name(hosts):
    return sorted(hosts)[0]


def _wait_for_worker_nodes_to_start_sshd(hosts, interval=1, timeout_in_seconds=180):
    with timeout(seconds=timeout_in_seconds):
        while hosts:
            print("hosts that aren't SSHable yet: %s", str(hosts))
            for host in hosts:
                ssh_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                if _can_connect(host, 22, ssh_socket):
                    hosts.remove(host)
            time.sleep(interval)


def _can_connect(host, port, s):
    try:
        print("testing connection to host %s", host)
        s.connect((host, port))
        s.close()
        print("can connect to host %s", host)
        return True
    except socket.error:
        print("can't connect to host %s", host)
        return False

class TimeoutError(Exception):
    pass

@contextmanager
def timeout(seconds=0, minutes=0, hours=0):
    """
    Add a signal-based timeout to any block of code.
    If multiple time units are specified, they will be added together to determine time limit.
    Usage:
    with timeout(seconds=5):
        my_slow_function(...)
    Args:
        - seconds: The time limit, in seconds.
        - minutes: The time limit, in minutes.
        - hours: The time limit, in hours.
    """

    limit = seconds + 60 * minutes + 3600 * hours

    def handler(signum, frame):  # pylint: disable=W0613
        raise TimeoutError('timed out after {} seconds'.format(limit))

    try:
        signal.signal(signal.SIGALRM, handler)
        signal.setitimer(signal.ITIMER_REAL, limit)
        yield
    finally:
        signal.alarm(0)

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

def _get_training_world():

    """
    Calculates number of devices in Sagemaker distributed cluster
    """

    # Get params of Sagemaker distributed cluster from predefined env variables
    num_gpus = int(os.environ["SM_NUM_GPUS"])
    num_cpus = int(os.environ["SM_NUM_CPUS"])
    hosts = json.loads(os.environ["SM_HOSTS"])
    current_host = os.environ["SM_CURRENT_HOST"]

    # Define PyTorch training world
    world = {}
    world["number_of_processes"] = num_gpus
    world["number_of_machines"] = len(hosts)
    world["master_addr"] = hosts[0]
    world["current_host"] = current_host
    world["master_port"] = "55555"

    return world

def _construct_host_file() -> str:
    """constructs MPI-compatible hostfile with all nodes 
     and number of GPU devices"""
    hostfile=os.path.join(os.environ['SM_INPUT_CONFIG_DIR'],'hostfile')
    if os.environ.get('SM_HOSTS', None) is not None:
        hosts =  json.loads(os.environ['SM_HOSTS'])
    else:
        hosts = ["localhost"]
    
    # replace current hosts with 'localhost'
    # TODO: remove after testing. Should work without this tweak in case of MPI
#     hosts[hosts.index(os.environ['SM_CURRENT_HOST'])]='127.0.0.1'
    print(f"list of hosts:{hosts} used to construct hostfile config")
        
    if len(hosts)==1:
        # for single node training
        # deepspeed will automatically use all local resources
        # no need to create host files
        logger.debug("Single node training, skipping hostfile creation.")
        return ""
    
    with open(hostfile,'a') as f:
        for h in hosts:
            f.write(f"{h} slots={os.environ['SM_NUM_GPUS']}\n")

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


def worker_routine(proccess_id_string, worker):
    """
    This method is executed on workers. 
    It waits for 60 seconds and then checks if training processes are spawned up.
    """
    
    print("Inside worker routine")

    training_process_started = False
    
    while True:
        time.sleep(60)
                
        training_process_ps = subprocess.check_output('ps -elf | grep "{}"'.format(proccess_id_string), encoding='utf-8', shell=True)
        training_process_count = subprocess.check_output('ps -elf | grep "{}" | wc -l'.format(proccess_id_string), encoding='utf-8', shell=True)
        training_process_count_str = training_process_count.replace("\n", "").strip()
        training_process_count = int(training_process_count_str) - 2
        training_process_running = training_process_count > 0
        if training_process_started:
            print('training processes running: {}'.format(training_process_count))
            if not training_process_running:
                print('Worker {} training completed.'.format(worker))
                time.sleep(5)
                sys.exit(0)

        if not training_process_started:
            if training_process_running:
                training_process_started = True
            else:
                print('Worker {} exiting: training not started in 60 seconds.'.format(worker))
                sys.exit(1)



def main():
    
    common_setup()
    
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
    

    # constructing script launcher based on
    # https://www.deepspeed.ai/getting-started/#launching-deepspeed-training
    train_script = os.path.join(os.environ['SM_MODULE_DIR'], args.train_script)
    hostfile = _construct_host_file()
    deepspeed_config = os.path.join(os.environ['SM_MODULE_DIR'], args.deepspeed_config_file)
    world = _get_training_world()
    
    if not _is_master_host():
        # worker nodes do nothing as deepspeed launcher
        # will distribute the tasks
        worker_routine(train_script, world["current_host"])
        logger.info(f"Worker {world['current_host']} has completed")
        return
    
    cmd = ["deepspeed"]
    if hostfile:
        cmd.append(f"--hostfile={hostfile}")
    #https://github.com/microsoft/DeepSpeed/blob/c5e4264186c321321858fb46d75c328fdc908ddb/deepspeed/launcher/runner.py#L95
    cmd.append("--launcher=openmpi")
    cmd.append("--launcher_args='--allow-run-as-root --display-map --tag-output'")
    cmd.append(f"--master_addr={world['master_addr']}")
    cmd.append(f"--master_port={world['master_port']}")
    cmd.append(train_script)
    cmd.append("--deepspeed-flag")
    cmd.append(f"--config-file={deepspeed_config}")
    # TODO: reconsider this feature adding all arguments which were not parsed by launcher argparse
#     cmd.append(unknown)
    logger.info(f"Following command line will be launched on master:{cmd}")
    
    # update current_env list
    env_vars = os.environ.copy()
    env_vars['NCCL_DEBUG']='DEBUG'
    env_vars['MASTER_PORT']='55555'
    logger.info(f"Env variables to be distributed across nodes: {env_vars}")

    process = subprocess.Popen(cmd, env=env_vars, stderr=subprocess.STDOUT, stdout=subprocess.PIPE)
    
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