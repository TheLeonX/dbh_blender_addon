"""Export-local image verification, overlapped with mesh work (no RNA workers)."""
import hashlib
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import numpy as np


class ExportHashes:
    def __init__(self,budget=256*1024*1024,workers=4):
        self.budget=budget;self.workers=workers;self.memory=0
        self.todo=deque();self.pending={};self.results={}
        self.pool=ThreadPoolExecutor(max_workers=workers,thread_name_prefix='DBH-export-hash')

    def __enter__(self):return self

    def __exit__(self,*error):
        self.pool.shutdown(wait=True,cancel_futures=True)
        self.pending.clear();self.todo.clear();self.memory=0

    @staticmethod
    def fingerprint(size,pixels):
        digest=hashlib.sha256(str(size).encode());digest.update(memoryview(pixels))
        return digest.hexdigest()

    def start(self,images):
        unique={image.as_pointer():image for image in images if image is not None}
        self.todo.extend(unique.items());self.pump()

    def pump(self):
        # All image reads, pointers and dimensions are accessed on the caller's
        # Blender thread. The workers receive only size tuples and float arrays.
        for key,(future,size) in list(self.pending.items()):
            if future.done():
                self.results[key]=future.result();self.memory-=size;del self.pending[key]
        while self.todo and len(self.pending)<self.workers:
            key,image=self.todo[0];count=len(image.pixels)
            if self.pending and self.memory+count*4>self.budget:break
            self.todo.popleft()
            pixels=np.empty(count,np.float32);image.pixels.foreach_get(pixels)
            future=self.pool.submit(self.fingerprint,tuple(image.size),pixels)
            self.pending[key]=(future,pixels.nbytes);self.memory+=pixels.nbytes

    def finish(self):
        while self.todo or self.pending:
            self.pump()
            if self.pending:
                # Wait on one ordered item, then drain and refill; never hold
                # more than four snapshots or the bounded image-memory budget.
                next(iter(self.pending.values()))[0].result()
        return self.results

    def get(self,image):
        key=image.as_pointer()
        # A native hair material can request its digest during the material
        # pass. Prioritize it without waiting for every unrelated texture.
        queued=next((item for item in self.todo if item[0]==key),None)
        if queued is not None:self.todo.remove(queued);self.todo.appendleft(queued)
        while key not in self.results and (key in self.pending or queued is not None):
            self.pump()
            if key in self.results:break
            if self.pending:self.pending.get(key,next(iter(self.pending.values())))[0].result()
        if key not in self.results:
            from .material_ui import pixel_hash
            self.results[key]=pixel_hash(image)
        return self.results[key]
