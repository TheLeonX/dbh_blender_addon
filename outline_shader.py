"""Backface rejection in every fragment pass of the verified outline carrier.

The carrier already declares FrontFacing in each Vulkan entry interface. Keep
that interface, QDIF and byte counts; add one structured selection/kill, funded
by debug names. Depth/shadow and velocity must reject the same hull faces as
color, otherwise the expanded front shell would occlude the original surface.
"""
import struct

EXTRA_PROFILES={
 (2,2):'c0aabc6b8626b5b6b3113b0c1beee11403be11ec9423d32669ac61dadde5146d',
 (2,3):'855dd8a7103d9e776219b8c8626bcad915271f5110717c2ad005e3ad9bb56106',
 (4,2):'e6ef3c6af5a5cf869f2746516909b198b2fa01c590ba3f024ca7a3687812ee69',
 (4,3):'9aa90145898c04fac8ab5825a3730082c65285297ef5bc2456719713b3abe9f5',
}


def detect_outline(raw):return b'// DBH_OUTLINE_V1\n' in raw[:230000]


def detect_face_mode(raw):
    import re
    match=re.search(rb'// DBH_FACE_CULL_V1 (FRONT|BACK)\n',raw[:230000])
    return match.group(1).decode() if match else 'BOTH'


def glsl(chunk,visible='FRONT',outline=True):
    if visible not in ('FRONT','BACK'):raise ValueError('Choose visible front or back faces')
    anchor=b'void main() {'
    if chunk.count(anchor)!=1 or chunk[-1:]!=b'\0':raise ValueError('Unknown outline GLSL entry point')
    rejection=b'!gl_FrontFacing' if visible=='FRONT' else b'gl_FrontFacing'
    chunk=chunk.replace(anchor,anchor+b'\n if ('+rejection+b') { discard; }\n')
    marker=b'// DBH_OUTLINE_V1' if outline else b'// DBH_FACE_CULL_V1 '+visible.encode()
    return chunk[:-1]+b'\n'+marker+b'\n\0'


def spirv(chunk,visible='FRONT'):
    if visible not in ('FRONT','BACK'):raise ValueError('Choose visible front or back faces')
    from .cel_shader import _instructions
    if chunk[:4]!=b'QDIF' or chunk[3944:3948]!=b'\x03\x02\x23\x07':raise ValueError('Unknown outline QDIF')
    words=list(struct.unpack('<'+'I'*((len(chunk)-3944)//4),chunk[3944:]))
    ins=list(_instructions(words))
    fronts=[w[1] for w in ins if w[0]&65535==71 and w[2:]==[11,17]]
    booleans=[w[1] for w in ins if w[0]&65535==20]
    entries=[w for w in ins if w[0]&65535==15]
    if len(fronts)!=1 or len(booleans)!=1 or len(entries)!=1 or entries[0][1]!=4:
        raise ValueError('Outline fragment FrontFacing interface mismatch')
    if any(w[0]&65535==16 and w[2]==9 for w in ins):raise ValueError('Early fragment tests are unsafe for outline discard')
    main=entries[0][2]
    function=next(i for i,w in enumerate(ins) if w[0]&65535==54 and w[2]==main)
    entry=next(i for i in range(function+1,len(ins)) if ins[i][0]&65535==248)
    label=ins[entry][1];at=entry+1
    while ins[at][0]&65535 in (59,8,317):at+=1
    condition,merge,killed=words[3],words[3]+1,words[3]+2
    def op(code,*args):return [((len(args)+1)<<16)|code,*args]
    branch=(condition,merge,killed) if visible=='FRONT' else (condition,killed,merge)
    body=(op(61,booleans[0],condition,fronts[0])+op(247,merge,0)+op(250,*branch)
          +op(248,killed)+op(252)+op(248,merge))
    removed=set();freed=0
    for i in reversed(range(len(ins))):
        if ins[i][0]&65535 in (5,6):
            removed.add(i);freed+=len(ins[i])
            if freed>=len(body):break
    if freed<len(body):raise ValueError('Not enough debug space for outline face rejection')
    out=words[:5];out[3]+=3
    for i,w in enumerate(ins):
        if i in removed:continue
        if i==at:out+=[1<<16]*(freed-len(body))+body
        if w[0]&65535==245:
            w=w.copy()
            for j in range(4,len(w),2):
                if w[j]==label:w[j]=merge
        out+=w
    if len(out)!=len(words):raise ValueError('Outline changed wrapped shader size')
    return chunk[:3944]+struct.pack('<'+'I'*len(out),*out)
