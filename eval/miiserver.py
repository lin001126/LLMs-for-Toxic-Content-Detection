# import mii
# from datetime import datetime, timedelta
# import os
# import json
# import pandas as pd
# from tqdm import tqdm
# import time
# from transformers import AutoTokenizer
# import pandas as pd
# client = mii.serve(
#     "/hpc2hdd/home/jzhao815/model/qwen4b_llamapro/checkpoint-700",
#     deployment_name="toxic",
#     max_length=1024,
#     enable_restful_api=True,
#     restful_api_port=28080,
# )
import mii
mii.terminate("toxic")