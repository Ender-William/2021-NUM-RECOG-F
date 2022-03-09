import os
import shutil


def custom_rmdir(rootdir):
    filelist=os.listdir(rootdir)
    for f in filelist:
      filepath = os.path.join( rootdir, f )
      if os.path.isfile(filepath):
        os.remove(filepath)
        #print(filepath+" removed!")
      elif os.path.isdir(filepath):
        shutil.rmtree(filepath,True)
        #print("dir "+filepath+" removed!")
