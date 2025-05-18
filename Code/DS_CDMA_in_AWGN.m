clear all;
close all;

% Generate random message
msg = round(rand(1, 1000)); % 1000-bit message

% ----- Simple Convolutional Encoder (rate 1/2, memory 2) -----
% Trellis: constraint length = 3, generators = [6 7] in octal
G1 = [1 1 0]; % binary for 6
G2 = [1 1 1]; % binary for 7
K = length(G1);
msg_padded = [zeros(1,K-1) msg]; % zero-padding for shift register
len_msg = length(msg);
encoded = [];

for i = 1:len_msg
    reg = msg_padded(i:i+K-1);
    out1 = mod(sum(G1 .* reg), 2);
    out2 = mod(sum(G2 .* reg), 2);
    encoded = [encoded out1 out2];
end

% ----- Map binary to bipolar NRZ format -----
user = encoded;
user(user==0) = -1;

% ----- BPSK Parameters -----
fc = 5000;              % Carrier frequency (Hz)
eb = 0.5;               % Energy per bit
bitrate = 1000;         % Bitrate (bps)
tb = 1 / bitrate;       % Bit duration
chiprate = 10000;       % Chip rate (chips/sec)
tc = 1 / chiprate;      % Chip duration

length_user = length(user);
t = tc:tc:tb*length_user;

% ----- Generate Baseband Signal -----
basebandsig = repelem(user, tb/tc);
figure(1)
stairs(t(1:800), basebandsig(1:800))
xlabel('Time (sec)')
ylabel('Binary value')
set(gca,'ytick',[-1 1])
title('Baseband signal for a single user')

% ----- BPSK Modulation -----
bpskmod = [];
for i = 1:length_user
    tbit = tc:tc:tb;
    bpskmod = [bpskmod sqrt(2*eb)*user(i)*cos(2*pi*fc*tbit)];
end

% ----- Frequency Domain Analysis -----
number = length(t);
spectrum = abs(fft(bpskmod));
sampling_frequency = 2 * fc;
sampling_interval = 1.0 / sampling_frequency;
for i = 1:number
    frequency(i) = (1.0/(number*sampling_interval)) * i;
end
figure(2)
plot(frequency, spectrum)
title('Frequency Domain of BPSK modulated signal')
xlabel('Frequency (Hz)')
ylabel('Magnitude')
grid on

% ----- PN Sequence Generation -----
seed = [1 -1 1 -1]; % initial bipolar seed
pn = [];
for i = 1:length_user
    for j = 1:10
        pn = [pn seed(4)];
        if seed(4) == seed(3)
            temp = -1;
        else
            temp = 1;
        end
        seed(4) = seed(3);
        seed(3) = seed(2);
        seed(2) = seed(1);
        seed(1) = temp;
    end
end

% ----- PN Upsampling -----
pnupsampled = repelem(pn, tb/(10*tc));

% ----- Spread Signal -----
sigtx = bpskmod .* pnupsampled;
figure(3)
plot(t(1:200), sigtx(1:200))
title('Transmitted DS-CDMA Signal')
xlabel('Time (sec)')
ylabel('Amplitude')
grid on

% ----- BER Simulation over AWGN -----
snr_in_dBs = 0:1:10;
ber = zeros(size(snr_in_dBs));

for m = 1:length(snr_in_dBs)
    % Add AWGN (manual)
    Ps = mean(sigtx.^2);               % Signal power
    SNR_linear = 10^(snr_in_dBs(m)/10);
    noise_power = Ps / SNR_linear;
    noise = sqrt(noise_power) * randn(1, length(sigtx));
    noisy_signal = sigtx + noise;

    % Despread
    rx = noisy_signal .* pnupsampled;

    % Demodulation
    demodcar = [];
    for i = 1:length_user
        tbit = tc:tc:tb;
        demodcar = [demodcar sqrt(2*eb)*cos(2*pi*fc*tbit)];
    end
    bpskdemod = rx .* demodcar;

    % Integrate & Dump
    sum_val = zeros(1, length_user);
    for i = 1:length_user
        sum_val(i) = sum(bpskdemod((i-1)*10 + 1 : i*10));
    end
    rxbits = sum_val > 0; % Hard decision

    % BER Calculation (no biterr or vitdec)
    txbits = encoded == 1;  % Original bits after convolutional encoder
    errors = sum(rxbits ~= txbits);
    ber(m) = errors / length_user;

    fprintf('SNR = %d dB, BER = %.5f\n', snr_in_dBs(m), ber(m));
end

% ----- Plot BER -----
figure(4)
plot(snr_in_dBs, ber, '-o')
xlabel('SNR (dB)')
ylabel('Bit Error Rate (BER)')
title('BER vs SNR for Coded DS-CDMA in AWGN')
grid on
legend('Simulated BER')
