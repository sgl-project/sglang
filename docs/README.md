# SGLang Documentation
This is the documentation repository for SGLang. It is auto-generated from https://github.com/sgl-project/sglang/tree/main/docs.

## Build the documentation website

### Dependency
```
pip install -r requirements.txt
```

### Build
```
make html
```

### Clean
To remove all generated files:
```
make clean
```

### Serve (preview)
Run an HTTP server and visit http://localhost:8000 in your browser.

```
python3 -m http.server --d _build/html
```

### Deploy
Clone [sgl-project.github.io](https://github.com/sgl-project/sgl-project.github.io) and make sure you have write access.

```bash
export DOC_SITE_PATH=../../sgl-project.github.io   # update this with your path
python3 deploy.py
```
