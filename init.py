from utils import init_workers, get_active_worker_ips

worker_ips = get_active_worker_ips()

init_workers(worker_ips)