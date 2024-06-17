import numpy as np
import skrf as rf
import scipy.signal as sig
def compute_multi(Thru,FEXT,NEXT,Data_Rate,A_ft,A_nt):
    ## init input data
    fb = Data_Rate * 1e9
    ft = fb * 4  # 4?
    fr = 0.75 * fb  # 0.75?
    f_range = Data_Rate * 1e9
    Ant = A_nt
    Aft = A_ft

    [ILD,FOM_ILD,IL_fitted,f_num,IL,or_impulse, impulse] = compute_sigle(Thru,Data_Rate)

    Thru.se2gmm(p=2)
    df = Thru.frequency.df_scaled[0] * 1e9
    freq = Thru.f

    pwf_value = PWF(freq, fb, ft, fr)
    Wnt = Ant ** 2 / fb * pwf_value
    Wft = Aft ** 2 / fb * pwf_value
    delta_n = 0
    delta_f = 0
    MDNEXT = np.zeros(np.size(IL))
    MDFEXT = np.zeros(np.size(IL))
    PSXT = np.zeros(np.size(IL))

    for F in FEXT:
        F.se2gmm(p=2)
        MDFEXT = np.sqrt(MDFEXT ** 2 + abs(F.s[:, 1, 0]) ** 2)
        PSXT = np.sqrt(PSXT ** 2 + abs(F.s[:, 1, 0]) ** 2)
    for N in NEXT:
        N.se2gmm(p=2)
        MDNEXT = np.sqrt(MDNEXT ** 2 + abs(N.s[:, 1, 0]) ** 2)
        PSXT = np.sqrt(PSXT ** 2 + abs(N.s[:, 1, 0]) ** 2)

    delta_n = np.sqrt(2 * df * np.sum(Wnt * abs(MDNEXT[0:f_num]) ** 2))
    delta_f = np.sqrt(2 * df * np.sum(Wft * abs(MDFEXT[0:f_num]) ** 2))
    ICN = np.sqrt(2 * df / fb * np.sum(pwf_value * PSXT ** 2)) / 2  ## Unit:V
    scale = Ant / Aft
    ICR = -20 * np.log10(abs(PSXT * scale)) + 20 * np.log10(abs(IL))

    return ILD,FOM_ILD,IL_fitted,ICN,ICR,f_num,IL,or_impulse, impulse

def compute_sigle(Thru,Data_Rate):
    fb = Data_Rate * 1e9
    ft = fb * 4  # 4?
    fr = 0.75 * fb  # 0.75?
    f_range = Data_Rate* 1e9
    Thru.se2gmm(p=2)
    IL = Thru.s[:, 1, 0]
        # data init
    [or_impulse, impulse] = s21_to_impulse_DC(IL)
    df = Thru.frequency.df_scaled[0] * 1e9
    f_num = int(f_range / df)
    f = Thru.f

    # fit the IL
    IL_fitted = get_ILD_fitted(IL[0:f_num], f[0:f_num])
    # calculate ILD
    ILD = 20 * np.log10(abs(IL[0:f_num])) - 20 * np.log10(abs(IL_fitted))
    # calculate FOM_ILD
    pwf_value = PWF(f, fb, ft, fr)

    FOM_ILD = np.sqrt(np.average(pwf_value[0:f_num] * np.power(ILD, 2)))  # 93A-56
    
    return ILD,FOM_ILD,IL_fitted,f_num,IL,or_impulse, impulse

def PWF(f, fb, ft, fr):
    PWF_data = np.power(np.sinc(f / fb), 2)
    PWF_trf = np.power(1 + np.power(f / ft, 4), -1)
    PWF_rx = np.power(1 + np.power(f / fr, 8), -1)
    PWF = PWF_data * PWF_trf * PWF_rx  # 93A-57
    return PWF

def get_ILD_fitted(sdd21, faxis_f2):
    # used for FD IL fitting
    # sdd21 us a complex insertion loss
    F = np.column_stack([
        np.ones(len(faxis_f2)) * sdd21,
        np.sqrt(faxis_f2) * sdd21,
        faxis_f2 * sdd21,
        faxis_f2 ** 2 * sdd21
    ])  # 93A-52
    np.seterr(divide='ignore', invalid='ignore')
    unwraplog = np.log(np.abs(sdd21)) + 1j * np.unwrap(np.angle(sdd21))
    L = sdd21 * unwraplog  # 93A-53
    alpha = np.linalg.inv(F.T.conjugate() @ F) @ F.T.conjugate() @ L  # 93A-54
    efit = alpha[0] + alpha[1] * np.sqrt(faxis_f2) + alpha[2] * faxis_f2 + alpha[3] * faxis_f2 ** 2  # 93A-51
    FIT = np.transpose(np.exp(efit))
    return FIT

def s21_to_impulse_DC(IL, EC_PULSE_TOL=0.01, EC_DIFF_TOL=1e-3, EC_REL_TOL=1e-2,impulse_response_truncation_threshold=1e-3,ENFORCE_CAUSALITY=1):
    # Creates a time-domain impulse response from frequency-domain IL data.
    # IL does not need to have DC but a corresponding frequency array
    # (freq_array) is required.
    #
    # Causality is imposed using the Alternating Projections Method. See also:
    # Quatieri and Oppenheim, "Iterative Techniques for Minimum Phase Signal
    # Reconstruction from Phase or Magnitude", IEEE Trans. ASSP-29, December
    # 1981 (http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=1163714)

    IL_symmetric = np.concatenate((IL[:-1], np.zeros(1), np.flipud(np.conj(IL[1:-1]))), axis=0)
    impulse_response = np.real(np.fft.ifft(IL_symmetric))
    L = len(impulse_response)

    original_impulse_response = impulse_response
    # Correct non-causal effects frequently caused by extrapolation of IL
    # Assumption: peak of impulse_response is in the first half, i.e. not anti-causal
    abs_ir = np.abs(impulse_response)
    a = np.where(abs_ir[:L // 2] > np.max(abs_ir[:L // 2]) * EC_PULSE_TOL)[0]
    start_ind = a[0]

    err = np.inf
    while not np.all(impulse_response == 0):
        impulse_response[:start_ind] = 0
        impulse_response[int(np.floor(L / 2)):] = 0
        IL_modified = np.abs(IL_symmetric) * np.exp(1j * np.angle(np.fft.fft(impulse_response)))
        ir_modified = np.real(np.fft.ifft(IL_modified))
        delta = np.abs(impulse_response - ir_modified)

        err_prev = err
        err = np.max(delta) / np.max(impulse_response)
        impulse_response = ir_modified
        if err < EC_REL_TOL or abs(err_prev - err) < EC_DIFF_TOL:
            break

    causality_correction_dB = 20 * np.log10(
        np.linalg.norm(impulse_response - original_impulse_response) / np.linalg.norm(impulse_response))

    if not ENFORCE_CAUSALITY:
        impulse_response = original_impulse_response

    # truncate final samples smaller than 1e-3 of the peak
    ir_peak = np.max(abs(impulse_response))
    ir_last = np.where(np.abs(impulse_response) > ir_peak * impulse_response_truncation_threshold)[0][-1]
    voltage = impulse_response[0:ir_last+1]
    return original_impulse_response, voltage

def s21_to_impulse_DC2(IL, freq_array, time_step, OP):
    ILin = IL
    fmax = 1 / time_step / 2
    freq_step = (freq_array[2] - freq_array[1]) / 1
    fout = np.arange(0, round(fmax / freq_step) * fmax, 1 / round(fmax / freq_step))
    #IL = interp_Sparam(ILin, freq_array, fout, OP.interp_sparam_mag, OP.interp_sparam_phase, OP)
    
    IL_nan = np.isnan(IL)
    for i in range(len(IL)):
        if IL_nan[i]:
            IL[i] = IL[i-1]
    
    IL = IL.reshape(-1, 1)
    IL_symmetric = np.concatenate((IL[:-1], np.zeros((1, 1)), np.flipud(np.conj(IL[1:-1]))))
    impulse_response = np.real(np.fft.ifft(IL_symmetric))
    L = len(impulse_response)
    t_base = np.arange(L) / (freq_step * L)
    
    original_impulse_response = impulse_response
    abs_ir = np.abs(impulse_response)
    a = np.where(abs_ir[:L//2] > np.max(abs_ir[:L//2]) * OP.EC_PULSE_TOL)[0]
    start_ind = a[0]
    
    err = np.inf
    while not np.all(impulse_response == 0):
        impulse_response[:start_ind] = 0
        impulse_response[L//2:] = 0
        IL_modified = abs(IL_symmetric) * np.exp(1j * np.angle(np.fft.fft(impulse_response)))
        ir_modified = np.real(np.fft.ifft(IL_modified))
        delta = np.abs(impulse_response - ir_modified)
        
        err_prev = err
        err = np.max(delta) / np.max(impulse_response)
        if err < OP.EC_REL_TOL or np.abs(err_prev - err) < OP.EC_DIFF_TOL:
            break
        
        impulse_response = ir_modified
    
    causality_correction_dB = 20 * np.log10(np.linalg.norm(impulse_response - original_impulse_response) /
                                             np.linalg.norm(impulse_response))
    
    if not OP.ENFORCE_CAUSALITY:
        impulse_response = original_impulse_response
    
    ir_peak = np.max(np.abs(impulse_response))
    ir_last = np.where(np.abs(impulse_response) > ir_peak * OP.impulse_response_truncation_threshold)[0][-1]
    
    voltage = impulse_response[:ir_last]
    t_base = t_base[:ir_last]
    
    truncation_dB = 20 * np.log10(np.linalg.norm(impulse_response[ir_last+1:]) / np.linalg.norm(voltage))
    
    return voltage, t_base, causality_correction_dB, truncation_dB

def impulse_to_pulse(imp_response,samples_per_ui):
    ## SBR
    pulse = sig.lfilter(np.ones(1,samples_per_ui), 1, imp_response)
    return pulse


def TX_FIR(imp_response,samples_per_ui,fir):
    num_tapes = np.size(fir)
    upsampled_txffe = np.zeros(samples_per_ui*(num_tapes-1)+1); # start with zeros everywhere
    upsampled_txffe[::samples_per_ui] = fir; # plant the coefficients in the desired locations
    TX_FIR_imp_response = filter(upsampled_txffe, 1, imp_response)
    return TX_FIR_imp_response


def get_RAW_FIR(H, f, OP, param):
    H_r = 1 / np.polyval([1, 2.613126, 3.414214, 2.613126, 1], 1j * f / (0.75 * param['fb']))
    
    if H.ndim > 1 and H.shape[1] != 1:
        H = np.squeeze(H)
    if H_r.ndim > 1 and H_r.shape[1] != 1:
        H_r = np.squeeze(H_r)
    
    H = H * H_r
    
    FIR, t, _, _ = s21_to_impulse_DC2(H, f, param['sample_dt'], OP)
    
    # 返回 FIR 和 t
    return FIR, t       

def Tukey_Window(f, param):
    fb = param['fb']
    fr = param['f_r'] * param['fb']
    fperiod = 2 * (fb - fr)
    
    H_tw = np.concatenate((np.ones(len(f[f < fr])),
                           0.5 * np.cos(2 * np.pi * (f[(f >= fr) & (f <= fb)] - fb) / fperiod - np.pi) + 0.5,
                           np.zeros(len(f[f > fb]))))
    
    H_tw = H_tw[:len(f)]
    
    return H_tw

def get_StepR(ir, param, cb_step, ZT):
    #ir = impulse response
    #t_base=time array with equal time steps
    #samp_UI = number of samples per UI for ir
    result = {}

    # t for debug
    t = (1/param["fb"]) / (param["samples_per_ui"]) * (range(len(ir)))

    if cb_step:
        Ag = 1
        dt = 1 / (param["fb"]) / (param["samples_per_ui"])
        edge_time = param["TR_TDR"] * 1e-9
        fedge = 1 / edge_time
        tedge = range(0, edge_time*2, dt)
        edge = Ag * (2 * np.cos(2 * np.pi * (tedge) * fedge / 16 - np.pi / 4) ** 2 - 1)
        drive_pulse = edge + [1] * (param["samples_per_ui"])
        pulse = sig.lfilter(drive_pulse, 1, ir)
    else:
        pulse = sig.lfilter([1] * len(ir), 1, ir)

    TDR_response = (1 + pulse) / (1 - pulse) * ZT * 2
    result["ZSR"] = TDR_response
    result["pulse"] = pulse
    return result

def get_PulseR(ir, param, cb_step, ZT):
    # ir = impulse response
    # t_base = time array with equal time steps
    # samp_UI = number of samples per UI for ir

    # t for debug
    t = np.arange(len(ir)) * (1/param['fb']) / param['samples_per_ui']

    if cb_step:
        Ag = 1
        dt = 1 / (param['fb'] * param['samples_per_ui'])
        edge_time = param['TR_TDR'] * 1e-9
        fedge = 1 / edge_time
        tedge = np.arange(0, edge_time * 2, dt)
        edge = Ag * (2 * np.cos(2 * np.pi * tedge * fedge / 16 - np.pi/4)**2 - 1)
        drive_pulse = np.concatenate([edge, np.ones(param['samples_per_ui'])])
        pulse = sig.lfilter(drive_pulse, 1, ir)
    else:
        pulse = sig.lfilter(np.ones(param['samples_per_ui']), 1, ir)

    PDR_response = (1 + pulse) / (1 - pulse) * ZT * 2
    result = {'PDR': PDR_response, 'pulse': pulse}
    return result

def get_pdf_from_sampled_signal(input_vector, L, BinSize, FAST_NOISE_CONV):
    if FAST_NOISE_CONV is None:
        FAST_NOISE_CONV = 0
    if max(input_vector) > BinSize:
        input_vector = input_vector[abs(input_vector) > BinSize]
    input_vector[abs(input_vector) < BinSize] = 0
    b = np.sign(input_vector)
    index = np.argsort(abs(input_vector))[::-1]
    input_vector = input_vector[index]
    input_vector = input_vector * b[index]
    if FAST_NOISE_CONV:
        sig_res = np.linalg.norm(input_vector[np.where(abs(input_vector) < .001)[0][0] + 1:])
        res_pdf = normal_dist(sig_res, 5, BinSize)
        input_vector = input_vector[:np.where(abs(input_vector) < .001)[0][0]]
    
    values = 2 * (0:L-1) / (L - 1) - 1
    prob = np.ones(L) / L

    pdf = d_cpdf(BinSize, 0, 1)

    for k in range(len(input_vector)):
        pdfn = d_cpdf(BinSize, abs(input_vector(k)) * values, prob)
        pdf = conv_fct(pdf, pdfn)

    if FAST_NOISE_CONV:
        pdf = conv_fct(pdf, res_pdf)

    return pdf

@staticmethod
def d_cpdf(binsize, values, probs):
    values = binsize * np.round(values / binsize)
    t = np.arange(min(values), max(values) + binsize, binsize)
  
    Min = min(values) // binsize
    y = np.zeros(len(t))
    for k in range(len(values)):
        bin = np.argmin(np.abs(t - values[k]))
        y[bin] += probs[k]

    y = y / np.sum(y)
    if any(~np.isreal(y)) or any(y < 0):
        raise ValueError('PDF must be real and nonnegative')

    support = np.where(y)[0]
    y = y[support[0]:support[-1]]
    Min += support[0]
    x = np.arange(Min, -Min + 1) * binsize

    pdf = {'x': x, 'y': y, 'BinSize': binsize}
    return pdf

@staticmethod
def conv_fct(p1, p2):
    if p1['BinSize'] != p2['BinSize']:
        raise ValueError('bin size must be equal')

    p = p1
    p['BinSize'] = p1['BinSize']
    p['Min'] = round(p1['Min'] + p2['Min'])
    p['y'] = np.convolve(p1['y'], p2['y'])
    p['x'] = np.arange(p['Min'] * p['BinSize'], -p['Min'] * p['BinSize'] + 1, p['BinSize'])

    return p

def normal_dist(sigma, nsigma, binsize):
    pdf = {}
    pdf['BinSize'] = binsize
    pdf['Min'] = -round(nsigma * sigma / binsize)
    pdf['x'] = np.arange(pdf['Min'], -pdf['Min']+1) * binsize
    pdf['y'] = np.exp(-pdf['x']**2 / (2 * sigma**2 + np.finfo(float).eps))
    pdf['y'] = pdf['y'] / np.sum(pdf['y'])
    
    return pdf


def get_TDR(sdd, TR_TDR, OP,param,ZT,nport):
    ## sdd is differential s-parameters structure (2 port assumed)
    ## input parameter structure for s parameters sdd--> sdd.Impedance, sdd.Frequencies, sdd.Parameters, sdd.NumPorts
    ## TDR_results.delay             pre t=0 delay for TDR... help with time domain responce quaility
    ## TDR_results.tdr               the TDR responce (ohms vs  TDR_results.t
    ## TDR_results.t                 starting at t=0
    ## TDR_results.tx_filter         transmitter filter vs TDR_results.f
    ## TDR_results.Rx_filter         receiver filter vs TDR_results.f
    ## TDR_results.f                 frequency for filter and s parameters
    ## TDR_results.ptdr_RL           reflection waveform from the pulse
    ## TDR_results.WC_ptdr_samples_t worst case time sample of the reflection pulse
    ## TDR_results.WC_ptdr_samples   worst case reflection samples of the reflection pulse
    ## TDR_results.ERL               reported effective return loss
    ## Only for Diff mode,only zin=100
    ## init fuction

    TDR_results = {
        "delay": [],
        "tdr": [],
        "t": [],
        "tx_filter":[],
        "rx_filter":[],
        "f":[],
        "ptdr_RL":[],
        "WC_ptdr_samples_t":[],
        "WC_ptdr_samples":[],
        "ERL":[],
        "RL":[],
        "avgZport":[],
        "ERLRMS":[]

    }
    


    db = lambda x: 20 * np.log10(abs(x))
    rms = lambda x: np.linalg.norm(x) / np.sqrt(len(x))
    TDR_RL = lambda Zin, Zout, s11, s12, s21, s22: (Zin**2 * s11 + Zin**2 * s22 + Zout**2 * s11 + Zout**2 * s22 + Zin**2 - Zout**2 + Zin * Zout * s11 * 2.0 - Zin * Zout * s22 * 2.0 + Zin**2 * s11 * s22 - Zin**2 * s12 * s21 - Zout**2 * s11 * s22 + Zout**2 * s12 * s21) / (Zin * Zout * 2.0 + Zin**2 * s11 + Zin**2 * s22 - Zout**2 * s11 - Zout**2 * s22 + Zin**2 + Zout**2 + Zin**2 * s11 * s22 - Zin**2 * s12 * s21 + Zout**2 * s11 * s22 - Zout**2 * s12 * s21 - Zin * Zout * s11 * s22 * 2.0 + Zin * Zout * s12 * s21 * 2.0)
    ## init param
    if 'TDR_duration' in OP:
        TDR_duration = OP['TDR_duration']
    else:
        TDR_duration = 5

    if 'DISPLAY_WINDOW' not in OP:
        OP['DISPLAY_WINDOW'] = 1

    RL = np.zeros(len(sdd.f))
    TDR_results['f'] = sdd.f                                           #result

    for i in range(len(TDR_results['f'])):
        RL[i] = TDR_RL(100,2*ZT,sdd.s[i,1,1],sdd.s[i,1,2],sdd[i,2,1],sdd[i,2,2])
    
    RL = np.squeeze(RL)
    f9 = TDR_results['f']/1e9
    tr = param.TR_TDR
    TDR_results['delay'] = 500e-12                                     #result

    try:
        maxtime = OP.N * param.ui
    except:
        maxtime = 2e-9
    
    if OP.N == 0:
        if sdd.NumPorts == 1:
            print("Warning for s2p files N must not be zero")
    else:
        fir4del, tu = get_RAW_FIR(sdd.s[2, 1, :], TDR_results['f'], OP, param)
        pix = np.argmax(fir4del)
        maxtime = tu[pix] * TDR_duration + TDR_results['delay']
        if maxtime > tu[-1]:
            maxtime = tu[-1]

# add delay 500 ps for TDR and 3 times Gaussnan transtion time
# (makes gausian edge somewhat causal)
    H_t = np.exp(-2*(np.pi*f9*(tr)/1.6832)**2) * np.exp(-1j*2*np.pi*f9*TDR_results['delay']/1e-9) * np.exp(-1j*2*np.pi*f9*tr*3)   #93A-46
    if 'cb_Guassian' not in OP:
        Use_gaussian = 1
    else:
        Use_gaussian = OP['cb_Guassian']
    
    if Use_gaussian:
        if H_t.ndim == 1:
            H_t = H_t[np.newaxis, :]  # 将H_t转换为行向量
        RLf = RL[:, np.newaxis] * H_t
    else:
        RLf = RL * np.exp(-1j * 2 * np.pi * f9 * TDR_results['delay'] / 1e-9)
    
    if 'cb_BesselThompson' in OP:
        OP['BesselThompson'] = OP['cb_BesselThompson']
    else:
        OP['BesselThompson'] = 0
    
    if 'f_r' in param:
        param['fb_BW_cutoff'] = param['f_r']
    
    if OP['BesselThompson']:
        a = np.poly( param['BTorder'] )
        acoef = np.flipud(a)
        H_bt = a[0] / np.polyval(acoef, (1j * TDR_results['f'] / (param['fb_BT_cutoff'] * param['fb'])))
    else:
        H_bt = np.ones(len(TDR_results['f']))
    
    
    if 'Butterworth' in OP:
        if OP['Butterworth']:
            H_bw = 1/np.polyval([1, 2.613126, 3.414214, 2.613126, 1], 1j*TDR_results['f']/(param['fb_BW_cutoff']*param['fb']))
        else:
            H_bw = np.ones(len(TDR_results['f']))
    else:
        H_bw = np.ones(len(TDR_results['f']))
    
    if param['Tukey_Window'] != 0:
        H_tw = Tukey_Window(TDR_results['f'], param)
    else:
        H_tw = np.ones(len(TDR_results['f']))
    
    if H_tw.ndim == 1:
        H_tw = H_tw.reshape(-1, 1)
    if H_bt.ndim == 1:
        H_bt = H_bt.reshape(-1, 1)
    if H_bw.ndim == 1:
        H_bw = H_bw.reshape(-1, 1)
    if RLf.ndim == 1:
        RLf = RLf.reshape(-1, 1)
    
    TDR_results['rx_filter'] = H_bt * H_bw * H_tw                #result
    RLf = RLf * TDR_results['rx_filter']
    TDR_results['tx_filter'] = H_t                #result
    

    [IR, t, causality_correction_dB, truncation_dB] = s21_to_impulse_DC2(RLf, TDR_results['f'], param.sample_dt,OP)

    tfx = param.tfx(nport)

    t = t-TDR_results['delay']
    tend = np.where(t >= maxtime)[0][0]
    if np.isnan(tend):
        tend = len(t)
    IR = IR[:tend]
    t = t[:tend]
    if np.isnan(tend):
        tend = len(t)
    tstart = np.where(t >= tr * 1e-9)[0][0]
    if np.isnan(tstart):
        tstart = 1
    
    if np.isnan(tend) or tstart >= tend:
        if np.isnan(tend) or tstart >= tend:
            # warndlg('TDR compuation not valid, try decreasing truncation tolerance, increasing samples, or adding a transmisson line','WrnTDR');
            pass
        tend = len(t)
        tstart = 1
    
    cb_step = 0

    ch = get_StepR(IR[tstart:tend],param,cb_step,ZT)

    TDR_results['tdr'] = ch.ZSR
    TDR_results['t'] = t[tstart:tend]

    PTDR = get_PulseR(IR[tstart:tend],param,cb_step,ZT)

    if OP['TDR'] or OP['PTDR']:
        try:
            tfstart = np.argmax(TDR_results['t'] >= 3 * tr * 1e-9)
            x = TDR_results['t'][tfstart:].squeeze()
            TDR_results['x'] = TDR_results['tdr'][:]
            y = TDR_results['tdr'][tfstart:].squeeze()
            TDR_results['y'] = TDR_results['t'][:]
            w = np.exp(-(x - x[0]) / OP['T_k'])  # weighting function
            TDR_results['avgZport'] = np.mean(y * w) / np.mean(w)
        except:
            TDR_results['avgZport'] = 0
            fit = np.zeros((1, 1))
            p = np.array([0, 0, 0, 0])
    TDR_results['RL'] = RL

    if OP.PTDR:
        RL_equiv = float('-inf')
        L = param.levels
        BinSize = OP.BinSize

        ntx = np.argmax(TDR_results.t >= tfx + 3*tr*1e-9)
        ndfex = np.argmax(TDR_results.t > (param.N_bx+1)*param.ui+tfx+3*tr*1e-9)
        tk = param.ui*1*(param.N_bx+1)+tfx+3*tr*1e-9
        
        if ndfex == 0:
            ndfex = len(TDR_results['t'])
        
        PTDR.pulse_orig = PTDR.pulse
        
        if param.Grr == 0:
            fctrx = (1 + param.rho_x) * param.rho_x * np.ones_like(PTDR.pulse_orig)
        elif param.Grr == 1:
            fctrx = np.ones_like(PTDR.pulse_orig)
        elif param.Grr == 2:
            fctrx = np.ones_like(PTDR.pulse_orig)
            
        Gloss = np.ones_like(TDR_results['t'])
        Grr = np.ones_like(TDR_results['t'])
        fctrx[:ntx] = 0
        
        for ii in range(ntx, ndfex):
            if param.N_bx > 0 and param.beta_x != 0:
                Gloss[ii] = 10**(param.beta_x*(TDR_results['t'][ii]-tk)/20)
            else:
                Gloss[ii] = 1
                
            x = (TDR_results['t'][ii] - tfx - 3*tr*1e-9) / param.ui
            
            if param.Grr == 0 or param.Grr == 1:
                Grr[ii] = (1 + param.rho_x) * param.rho_x * np.exp(-(x-1*param.N_bx-1)**2 / (1+param.N_bx)**2)
            elif param.Grr == 2:
                Grr[ii] = param.rho_x
                
            fctrx[ii] = Gloss[ii] * Grr[ii]
            
        PTDR.pulse = PTDR.pulse * fctrx
        
        FAST_NOISE_CONV = 0
        ERLRMS = np.sqrt(np.mean(np.square(PTDR.pulse)))
        
        for ki in range(1, param.samples_per_ui+1):
            progress = ki / param.samples_per_ui
            
            tps = PTDR.pulse[ki-1::param.samples_per_ui]
            
            if OP.RL_norm_test:
                rl_fom = np.linalg.norm(tps)
            else:
                testpdf = get_pdf_from_sampled_signal(tps, L, BinSize*10, FAST_NOISE_CONV)
                cdf_test = np.cumsum(testpdf.y)
                rl_test = -testpdf.x[np.argmax(cdf_test >= param.specBER)]
                rl_fom = rl_test
                
            if rl_fom > RL_equiv:
                RL_equiv = rl_fom
                best_ki = ki
                
            if not OP.RL_norm_test:
                best_erl = rl_test
                best_pdf = testpdf
                best_cdf = cdf_test
                
        if OP.RL_norm_test:
            tps = PTDR.pulse[best_ki-1::param.samples_per_ui]
            testpdf = get_pdf_from_sampled_signal(tps, L, BinSize*10, FAST_NOISE_CONV)
            cdf_test = np.cumsum(testpdf.y)
            best_erl = -testpdf.x[np.argmax(cdf_test >= param.specBER)]
            
        print()
        
        
        if 'best_ki' not in locals():
            best_ki = 1
        
        TDR_results['ptdr_RL'] = PTDR.pulse
        TDR_results['WC_ptdr_samples_t'] = TDR_results['t'][best_ki-1::param.samples_per_ui]
        TDR_results['WC_ptdr_samples'] = PTDR.pulse[best_ki-1::param.samples_per_ui]
        TDR_results['ERL'] = -db(best_erl)
        TDR_results['ERLRMS'] = -db(ERLRMS)
    return TDR_results


