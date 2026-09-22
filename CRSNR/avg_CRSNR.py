import os,numpy as np,soundfile as sf,matplotlib.pyplot as plt
from scipy.signal import stft,resample_poly

plt.rcParams["font.family"]="Times New Roman"

ships=[
"data/RORO_58.wav","data/Sailboat_68.wav",
"data/Cargo_3.wav","data/Cargo_6.wav",
"data/Cargo_55.wav","data/Cargo_107.wav",
"data/Passengers_111.wav","data/Passengers_164.wav",
"data/Tanker_119.wav","data/Tanker_133.wav",
"data/Tanker_140.wav","data/Tug_43.wav",
"data/Tug_47.wav"]
noises=[
"data/N1.wav","data/N2.wav","data/N3.wav",
"data/N4.wav","data/N5.wav","data/N6.wav"]

save="avg_CRSNR_results"
os.makedirs(save,exist_ok=True)

fs=16000;N=fs*30;hop=400;nfft=800;amax=20


def load(p):
    x,f=sf.read(p)
    if x.ndim>1:x=x[:,0]
    if f!=fs:x=resample_poly(x,fs,f)
    x=x[:N];x-=x.mean()
    return x/(np.sqrt(np.mean(x*x))+1e-12)


def cmc(x):
    _,_,Z=stft(
        x,fs=fs,nperseg=nfft,
        noverlap=nfft-hop,
        window="hann",
        boundary=None)
    Z=np.abs(Z[:400])
    C=np.abs(np.fft.rfft(
            Z*np.hanning(Z.shape[1]),
            axis=1))
    a=np.fft.rfftfreq(Z.shape[1],hop/fs)[1:]

    C=C[:,1:]
    idx=a<=amax
    return C[:,idx],a[idx]


def snr(C1,C2,a):
    return np.array([
        10*np.log10(
            (np.sum(C1[:,:i+1]**2)+1e-12)/
            (np.sum(C2[:,:i+1]**2)+1e-12))
        for i in range(len(a))])


def fc(C,a):
    e=C.sum(0)
    e/=e.sum()+1e-12
    return (a*e).sum()


for i,sp in enumerate(ships):

    name=os.path.splitext(os.path.basename(sp))[0]
    tag=chr(ord("a")+i)

    Cs,a=cmc(load(sp))
    center=fc(Cs,a)

    plt.figure(figsize=(3,4))
    plt.tick_params(
        axis="both",
        labelsize=12)

    for npth in noises:
        Cn, _ = cmc(load(npth))

        noise_id = os.path.splitext(
            os.path.basename(npth)
        )[0][1:]  # 去掉N

        plt.plot(
            a,
            snr(Cs, Cn, a),
            lw=1.5,
            label=f"Noise {noise_id}"
        )


    plt.xlabel("Cyclic frequency cutoff (Hz)",fontsize=12)

    plt.ylabel("Accumulated Relative CMC-SNR (dB)",fontsize=12)

    plt.title(f"({tag}) {name}",
        fontsize=12)

    plt.xlim(0,amax)

    plt.grid(ls="--",alpha=0.5)

    plt.legend(loc="upper right",fontsize=10)

    plt.tight_layout()

    plt.savefig(
        f"{save}/{name}_CMC_SNR.png",
        dpi=600,
        bbox_inches="tight")

    plt.close()

print("saved:",save)

