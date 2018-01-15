import zipfile
import os
from six.moves import urllib
dir_path = "data/glove.6B"
try:
  os.makedirs(dir_path, 0777)
except OSError:
	pass
def download_and_unzip(url_base, zip_name, *file_names):
  zip_path = os.path.join(dir_path, zip_name)
  url = url_base + zip_name
  print('downloading %s to %s' % (url, zip_path))
  urllib.request.urlretrieve(url, zip_path)
  out_paths = []
  with zipfile.ZipFile(zip_path, 'r') as f:
    for file_name in file_names:
      print('extracting %s' % file_name)
      out_paths.append(f.extract(file_name, path=dir_path))
  return out_paths

download_and_unzip("https://nlp.stanford.edu/data/","glove.6B.zip","glove.6B.100d.txt")
