import numpy as np

def compute_multi(Thru, FEXT, NEXT, Data_Rate, A_ft, A_nt):
    # Initialize input data
    fb = Data_Rate * 1e9
    ft = fb * 4
    fr = 0.75 * fb
    f_range = Data_Rate * 1e9
    ant_forward = A_nt
    ant_next = A_ft

    [ILD, FOM_ILD, IL_fitted, f_num, IL, or_impulse, impulse] = compute_single(Thru, Data_Rate)

    Thru.se2gmm(p=2)
    df = Thru.frequency.df_scaled[0] * 1e9
    freq = Thru.f

    pwf_value = PWF(freq, fb, ft, fr)
    Wnt = ant_forward ** 2 / fb * pwf_value
    Wft = ant_next ** 2 / fb * pwf_value
    delta_n = 0
    delta_f = 0
    MDNEXT = np.zeros(f_num)
    MDFEXT = np.zeros(f_num)
    PSXT = np.zeros(f_num)

    for F in FEXT:
        F.se2gmm(p=2)
        MDFEXT += np.abs(F.s[:, 1, 0]) ** 2
    for N in NEXT:
        N.se2gmm(p=2)
        MDNEXT += np.abs(N.s[:, 1, 0]) ** 2

    delta_n = np.sqrt(2 * df * np.sum(Wnt * MDNEXT))
    delta_f = np.sqrt(2 * df * np.sum(Wft * MDFEXT))
    ICN = np.sqrt(2 * df / fb * np.sum(pwf_value * PSXT ** 2)) / 2
    scale = ant_forward / ant_next
    ICR = -20 * np.log10(PSXT * scale) + 20 * np.log10(IL)

    return ILD, FOM_ILD, IL_fitted, ICN, ICR, f_num, IL, or_impulse, impulse

def compute_single(Thru, Data_Rate):
    fb = Data_Rate * 1e9
    ft = fb * 4
    fr = 0.75 * fb
    f_range = Data_Rate * 1e9
    Thru.se2gmm(p=2)
    IL = Thru.s[:, 1, 0]

    [or_impulse, impulse] = s21_to_impulse_DC(IL)
    f_num = int(f_range / df)
    f = Thru.f

    IL_fitted = get_ILD_fitted(IL[0:f_num], f[0:f_num])
    ILD = 20 * np.log10(abs(IL[0:f_num])) - 20 * np.log10(abs(IL_fitted))
    FOM_ILD = np.sqrt(np.average(pwf_value[0:f_num] * np.power(ILD, 2)))

    return ILD, FOM_ILD, IL_fitted, f_num, IL, or_impulse, impulse

def PWF(f, fb, ft, fr):
    PWF_data = np.power(np.sinc(f / fb), 2)
    PWF_trf = np.power(1 + np.power(f / ft, 4), -1)
    PWF_rx = np.power(1 + np.power(f / fr, 8), -1)
    PWF = PWF_data * PWF_trf * PWF_rx
    return PWF

def get_ILD_fitted(sdd21, faxis_f2):
    F = np.column_stack([
        np.ones(len(faxis_f2)) * sdd21,
        np.sqrt(faxis_f2) * sdd21,
        faxis_f2 * sdd21,
        faxis_f2 ** 2 * sdd21
    ])
    L = sdd21 * np.log(np.abs(sdd21)) + 1j * np.unwrap(np.angle(sdd21))
    alpha = np.linalg.inv(F.T.conjugate() @ F) @ F.T.conjugate() @ L
    efit = alpha[0] + alpha[1] * np.sqrt(faxis_f2) + alpha[2] * faxis_f2 + alpha[3] * faxis_f2 ** 2
    FIT = np.exp(efit)
    return FIT

def s21_to_impulse_DC2(IL, freq_array, time_step, OP):
    # 数据预处理与变量初始化
    fmax = 1 / time_step / 2
    freq_step = (freq_array[2] - freq_array[1]) / 1
    fout = np.arange(0, round(fmax / freq_step) * fmax, 1 / round(fmax / freq_step))
    
    ILin = np.nan_to_num(IL, copy=False)  # 使用numpy将NaN替换为前一个非NaN值
    
    IL = IL.reshape(-1, 1)
    IL_symmetric = np.concatenate((IL[:-1], np.zeros((1, 1)), np.flipud(np.conj(IL[1:-1]))))

    impulse_response = np.real(np.fft.ifft(IL_symmetric))
    L = len(impulse_response)
    t_base = np.arange(L) / (freq_step * L)

    original_impulse_response = impulse_response.copy()

    # 对于循环部分，这里假设OP.EC_PULSE_TOL、OP.EC_REL_TOL、OP.EC_DIFF_TOL均为固定阈值
    # 实际情况可能需要根据具体业务逻辑调整优化方式
    condition = np.abs(np.fft.fft(impulse_response)) > np.max(np.abs(np.fft.fft(impulse_response))) * OP.EC_PULSE_TOL
    start_ind = np.argmax(condition[:L//2])
    err = np.inf
    max_iterations = 1000  # 设置最大迭代次数以防止无限循环
    for _ in range(max_iterations):
        modified_mask = np.ones_like(impulse_response, dtype=bool)
        modified_mask[:start_ind] = False
        modified_mask[L//2:] = False
        IL_modified = abs(IL_symmetric) * np.exp(1j * np.angle(np.fft.fft(impulse_response[modified_mask])))
        
        ir_modified = np.real(np.fft.ifft(IL_modified))
        delta = np.abs(impulse_response - ir_modified)
        err_prev=err
        err = np.max(delta) / np.max(impulse_response)
        if err < OP.EC_REL_TOL or np.abs(err_prev - err) < OP.EC_DIFF_TOL:
            break
        
        impulse_response[modified_mask] = ir_modified[modified_mask]

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