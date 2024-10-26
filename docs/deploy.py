#!/usr/bin/python3

import os
from datetime import datetime


def run_cmd(cmd):
    print(cmd)
    os.system(cmd)


run_cmd("cd $DOC_SITE_PATH; git pull")

# (Optional) Remove old files
# run_cmd("rm -rf $ALPA_SITE_PATH/*")

run_cmd("cp -r _build/html/* $DOC_SITE_PATH")

cmd_message = f"Update {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
run_cmd(
    f"cd $DOC_SITE_PATH; git add .; git commit -m '{cmd_message}'; git push origin main"
)
