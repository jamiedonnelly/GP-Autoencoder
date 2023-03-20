import logging 
import os

# Conditional logger 
def logger(args,info):
    if args.rank==0:
        args.log.info(info)
    else:
        pass

# Create logger  
def get_logger(dir,filename):

    # Remove existing log if exists 
    filename = os.path.join(dir,filename)
    if filename in os.listdir(dir):
        os.system(f"rm '{filename}'")

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
