"""Read-only texture previews: PC DDS and QD JPEG-style texture blocks.

QD entropy order verified against game routines 140BB50A0 / 140BAAAF0 /
140BABCB0. Floating IDCT produces preview pixels, not bit-exact game BC data.
No external executables. NumPy is bundled with Blender.
"""
import math
import struct
import zlib
import numpy as np

ZIGZAG = tuple(bytes.fromhex('000108100902030a11182019120b04050c131a21283029221b140d06070e151c232a313839322b241d160f171e252c333a3b342d261f272e353c3d362f373e3f'))
QY = np.array([16,11,10,16,24,40,51,61,12,12,14,19,26,58,60,55,14,13,16,24,40,57,69,56,14,17,22,29,51,87,80,62,18,22,37,56,68,109,103,77,24,35,55,64,81,104,113,92,49,64,78,87,103,121,120,101,72,92,95,98,112,100,103,99]).reshape(8,8)
QC = np.array([17,18,24,47,99,99,99,99,18,21,26,66,99,99,99,99,24,26,56,99,99,99,99,99,47,66,99,99,99,99,99,99]+[99]*32).reshape(8,8)
IDCT = np.array([[.5 * (1 / math.sqrt(2) if u == 0 else 1) * math.cos((2*x+1)*u*math.pi/16) for u in range(8)] for x in range(8)])
DHT = [
 '0000010501010101010100000000000000000102030405060708090a0b',
 '100002010303020403050504040000017d01020300041105122131410613516107227114328191a1082342b1c11552d1f02433627282090a161718191a25262728292a3435363738393a434445464748494a535455565758595a636465666768696a737475767778797a838485868788898a92939495969798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9dae1e2e3e4e5e6e7e8e9eaf1f2f3f4f5f6f7f8f9fa',
 '0100030101010101010101010000000000000102030405060708090a0b',
 '1100020102040403040705040400010277000102031104052131061241510761711322328108144291a1b1c109233352f0156272d10a162434e125f11718191a262728292a35363738393a434445464748494a535455565758595a636465666768696a737475767778797a82838485868788898a92939495969798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9dae2e3e4e5e6e7e8e9eaf2f3f4f5f6f7f8f9fa']

def huffman(raw):
    raw = bytes.fromhex(raw); table = {}; code = 0; i = 17
    for n, count in enumerate(raw[1:17], 1):
        for _ in range(count):
            table[n, code] = raw[i]; code += 1; i += 1
        code <<= 1
    return table

TABLES = [huffman(raw) for raw in DHT]

class Bits:
    def __init__(self, raw): self.raw = raw; self.pos = 0
    def read(self, count):
        if self.pos + count > len(self.raw)*8: raise ValueError('Truncated QD entropy stream')
        result = 0
        for _ in range(count):
            result = (result << 1) | ((self.raw[self.pos >> 3] >> (7 - (self.pos & 7))) & 1)
            self.pos += 1
        return result
    def symbol(self, table):
        code = 0
        for n in range(1,17):
            code = (code << 1) | self.read(1)
            if (n,code) in table: return table[n,code]
        raise ValueError('Invalid QD Huffman code')
    def signed(self, n):
        value = self.read(n)
        return value - (1 << n) + 1 if n and value < (1 << (n-1)) else value

def decode_qd(raw):
    if len(raw)<52: raise ValueError('Truncated QD texture')
    version, fmt, quality, width, height = struct.unpack_from('<5H', raw)
    if version != 16 or fmt not in (1,3,7,15,17,19,21,22,23):
        raise ValueError(f'Unsupported QD preview format {fmt}')
    if not 0<width<=16384 or not 0<height<=16384: raise ValueError('Invalid QD dimensions')
    scale = 5000 // max(1,quality) if quality<50 else 2*(100-min(100,quality))
    qy = np.clip((QY*scale+50)//100,1,255)
    qc = np.clip((QC*scale+50)//100,1,255)
    qa = np.clip((QY*20+50)//100,1,255)
    size = 16 if fmt in (1,3,7) else 8
    output = np.zeros((height,width,4), dtype=np.uint8); output[:,:,3]=255
    count = struct.unpack_from('<I',raw,40)[0]
    if count>100000 or 52+count*16>len(raw): raise ValueError('Invalid QD chunk directory')
    # The reference wrapper keeps narrow raw grayscale images in an aligned
    # payload instead of entropy-coded tiles (e.g. 512 x 8 ramp textures).
    if count==1 and fmt==15:
        blocks,packed,decoded,offset=struct.unpack_from('<4I',raw,52)
        start=(52+16+15)&~15
        if blocks==0 and offset==0 and packed==decoded==width*height and len(raw)==start+packed:
            gray=np.frombuffer(raw[start:],np.uint8).reshape(height,width)
            output[:,:,:3]=gray[:,:,None]
            return width,height,output.tobytes()
    tile = 0
    for chunk in range(count):
        blocks, packed, decoded, offset = struct.unpack_from('<4I',raw,52+16*chunk)
        if offset+packed>len(raw) or blocks>100000: raise ValueError('Invalid QD chunk bounds')
        bits=Bits(raw[offset:offset+packed]); previous=[0,0,0,0]
        def plane(channel, chroma=False, alpha=False):
            dc,ac = TABLES[2:4] if chroma else TABLES[:2]
            coefficients=np.zeros(64)
            previous[channel]+=bits.signed(bits.symbol(dc));coefficients[0]=previous[channel]
            i=1
            while i<64:
                symbol=bits.symbol(ac);n=symbol&15;run=symbol>>4
                if not n:
                    if run==15: i+=16;continue
                    break
                i+=run
                if i>=64: raise ValueError('Invalid QD coefficient run')
                coefficients[ZIGZAG[i]]=bits.signed(n);i+=1
            quant = qa if alpha and fmt==3 else qc if chroma else qy
            return IDCT @ (coefficients.reshape(8,8)*quant) @ IDCT.T
        for _ in range(blocks):
            x=(tile % ((width+size-1)//size))*size;y=(tile//((width+size-1)//size))*size
            if y>=height: raise ValueError('Too many QD tiles')
            tile+=1
            alpha=None
            if fmt in (1,3,7):
                ys=[plane(0) for i in range(4)]
                lum=np.block([[ys[0],ys[1]],[ys[2],ys[3]]])
                # QD uses YCoCg, NOT JPEG YCbCr; no level shift of 128.
                co=plane(1,True).repeat(2,0).repeat(2,1)
                cg=plane(2,True).repeat(2,0).repeat(2,1)
                rgb=np.stack((lum+co-cg,lum+cg,lum-co-cg),axis=2)
                if fmt in (1,3):
                    aa=[plane(3,alpha=True) for i in range(4)]
                    alpha=np.block([[aa[0],aa[1]],[aa[2],aa[3]]])
            elif fmt==19: rgb=np.stack([plane(i) for i in range(3)],axis=2)
            elif fmt in (17,22,23): rgb=np.stack((plane(0),plane(1),np.zeros((8,8))),axis=2)
            else:
                gray=plane(0);rgb=np.stack((gray,gray,gray),axis=2)
            h=min(size,height-y);w=min(size,width-x)
            output[y:y+h,x:x+w,:3]=np.clip(np.rint(rgb[:h,:w]),0,255).astype(np.uint8)
            if alpha is not None: output[y:y+h,x:x+w,3]=np.clip(np.rint(alpha[:h,:w]),0,255).astype(np.uint8)
    if tile != ((width+size-1)//size)*((height+size-1)//size): raise ValueError('Incomplete QD image')
    return width,height,output.tobytes()


def alpha_block(raw):
    a,b=raw[:2];values=[a,b]
    if a>b: values += [((7-i)*a+i*b)//7 for i in range(1,7)]
    else: values += [((5-i)*a+i*b)//5 for i in range(1,5)]+[0,255]
    indices=int.from_bytes(raw[2:8],'little')
    return [values[(indices>>(3*i))&7] for i in range(16)]


def decode_gpu(w,h,data,fmt):
    if fmt==0:
        if len(data)<w*h*4:raise ValueError('Truncated RGBA8 texture')
        # 14067B150 maps engine format 0 to VkFormat 44/50 (BGRA8).
        a=np.frombuffer(data[:w*h*4],np.uint8).reshape(h,w,4)
        return w,h,a[:,:,(2,1,0,3)].tobytes()
    fourcc={5:b'DXT1',6:b'DXT3',7:b'DXT5',18:b'ATI1',19:b'ATI2'}.get(fmt)
    if fourcc:
        header=bytearray(128);header[:4]=b'DDS '
        struct.pack_into('<I',header,4,124);struct.pack_into('<II',header,12,h,w)
        header[84:88]=fourcc
        return decode_dds(bytes(header)+data)
    if fmt==11:  # R8, verified FILETEXT/raw grayscale resources.
        a=np.frombuffer(data[:w*h],np.uint8).reshape(h,w,1)
        return w,h,np.concatenate((np.repeat(a,3,2),np.full((h,w,1),255,np.uint8)),2).tobytes()
    raise ValueError(f'Unsupported GPU texture format {fmt}')


def decode_qd_raw(raw,gpu_format):
    if len(raw)<52: raise ValueError('Truncated QD raw texture')
    version,fmt,quality,w,h=struct.unpack_from('<5H',raw)
    if version!=16 or fmt!=255: raise ValueError('Not a QD raw texture')
    count=struct.unpack_from('<I',raw,40)[0]
    if not 0<count<=100000 or 52+16*count>len(raw): raise ValueError('Invalid QD raw directory')
    data=bytearray()
    for j in range(count):
        blocks,packed,decoded,offset=struct.unpack_from('<4I',raw,52+j*16)
        if decoded>16*1024*1024 or offset+packed>len(raw): raise ValueError('Invalid QD raw chunk')
        chunk=raw[offset:offset+packed]
        if packed==decoded:expanded=chunk
        else:
            inflater=zlib.decompressobj();expanded=inflater.decompress(chunk,decoded+1)
            if not inflater.eof or inflater.unconsumed_tail or len(expanded)!=decoded:
                raise ValueError('Invalid QD zlib chunk')
        data+=expanded
    return decode_gpu(w,h,bytes(data),gpu_format)

def decode_dds(raw):
    if len(raw)<128 or raw[:4]!=b'DDS ': raise ValueError('Not a DDS texture')
    h,w=struct.unpack_from('<2I',raw,12); fmt=raw[84:88]; start=128
    if not 0<w<=16384 or not 0<h<=16384: raise ValueError('Invalid DDS dimensions')
    if fmt==b'DX10':
        dxgi=struct.unpack_from('<I',raw,128)[0];start=148
        fmt={71:b'DXT1',72:b'DXT1',74:b'DXT3',75:b'DXT3',77:b'DXT5',78:b'DXT5',80:b'ATI1',83:b'ATI2',61:b'R8',28:b'RGBA',29:b'RGBA'}.get(dxgi)
    if fmt in (b'R8',b'RGBA'):
        channels=1 if fmt==b'R8' else 4
        pixels=np.frombuffer(raw[start:start+w*h*channels],dtype=np.uint8).reshape(h,w,channels)
        if channels==1: pixels=np.concatenate([np.repeat(pixels,3,2),np.full((h,w,1),255,np.uint8)],axis=2)
        return w,h,pixels.tobytes()
    if fmt not in (b'DXT1',b'DXT3',b'DXT5',b'ATI1',b'ATI2'): raise ValueError('Unsupported DDS format')
    blocksize=8 if fmt in (b'DXT1',b'ATI1') else 16
    expected=((w+3)//4)*((h+3)//4)*blocksize
    if len(raw)-start<expected: raise ValueError('Truncated DDS blocks')
    out=bytearray(w*h*4)
    for by in range(0,h,4):
        for bx in range(0,w,4):
            block=raw[start:start+blocksize];start+=blocksize
            if fmt in (b'ATI1',b'ATI2'):
                r=alpha_block(block);g=alpha_block(block[8:]) if fmt==b'ATI2' else r
                colors=[(r[i],g[i],0 if fmt==b'ATI2' else r[i],255) for i in range(16)]
            else:
                colorblock=block if fmt==b'DXT1' else block[8:]
                a,b,bits=struct.unpack('<HHI',colorblock)
                def rgb(v): return ((v>>11)*255//31,((v>>5)&63)*255//63,(v&31)*255//31,255)
                pal=[rgb(a),rgb(b)]
                if a>b or fmt!=b'DXT1':
                    pal += [tuple((2*pal[0][i]+pal[1][i])//3 for i in range(4)),tuple((pal[0][i]+2*pal[1][i])//3 for i in range(4))]
                else: pal += [tuple((pal[0][i]+pal[1][i])//2 for i in range(4)),(0,0,0,0)]
                colors=[pal[(bits>>(2*i))&3] for i in range(16)]
                if fmt!=b'DXT1':
                    alphas=alpha_block(block) if fmt==b'DXT5' else [(int.from_bytes(block[:8],'little')>>(4*i)&15)*17 for i in range(16)]
                    colors=[(*c[:3],alphas[i]) for i,c in enumerate(colors)]
            for i,color in enumerate(colors):
                x,y=bx+i%4,by+i//4
                if x<w and y<h: out[(y*w+x)*4:(y*w+x+1)*4]=bytes(color)
    return w,h,bytes(out)

def png_bytes(width,height,pixels):
    def chunk(kind,data): return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)
    rows=b''.join(b'\0'+pixels[y*width*4:(y+1)*width*4] for y in range(height))
    return b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>2I5B',width,height,8,6,0,0,0))+chunk(b'IDAT',zlib.compress(rows))+chunk(b'IEND',b'')
