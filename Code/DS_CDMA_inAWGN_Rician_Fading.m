clc; clear all; close all;

%% Message Generation
msg = randi([0 1], 1, 1000); % Random binary message

%% Custom Convolutional Encoder (1/2 rate, constraint length = 3)
g1 = [1 1 1]; % Generator polynomial: 7
g2 = [1 0 1]; % Generator polynomial: 5
msg_padded = [0 0 msg]; % Zero padding
user = [];

for i = 1:length(msg)
    shift_reg = msg_padded(i:i+2);
    user(end+1) = mod(sum(shift_reg .* g1), 2);
    user(end+1) = mod(sum(shift_reg .* g2), 2);
end

user(user == 0) = -1; % Bipolar NRZ format

%% System Parameters
fc = 5000;             % Carrier frequency in Hz
eb = 0.5;              % Energy per bit
bitrate = 1000;        % 1 Kbps
tb = 1 / bitrate;      % Bit duration
chiprate = 10000;      % Chip rate
tc = 1 / chiprate;     % Chip duration
length_user = length(user);
t = tc:tc:tb * length_user;

%% Baseband Signal
basebandsig = repelem(user, tb / tc);

figure(1)
stairs(t(1:800), basebandsig(1:800));
xlabel('Time (sec)');
ylabel('Binary value');
set(gca, 'ytick', [-1 1]);
title('Baseband signal for a single user');

%% BPSK Modulation
bpskmod = sqrt(2 * eb) * basebandsig .* cos(2 * pi * fc * t);

%% Frequency Spectrum
spectrum = abs(fft(bpskmod));
fs = 2 * fc;
freq = (0:length(t) - 1) * (fs / length(t));

figure(2)
plot(freq, spectrum);
title('Frequency Domain of BPSK signal');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
grid on;

%% PN Sequence Generation
seed = [1 -1 1 -1];
pn = [];

for i = 1:length_user
    for j = 1:10
        pn = [pn seed(4)];
        temp = (seed(4) ~= seed(3)) * 2 - 1;
        seed = [temp seed(1:3)];
    end
end

pnupsampled = repelem(pn, tb / (10 * tc));

%% Spread Signal
sigtx = bpskmod .* pnupsampled;

figure(3)
plot(t(1:200), sigtx(1:200));
title('Transmitted DS-CDMA Signal');
xlabel('Time (sec)');
ylabel('Amplitude');
grid on;

%% Rician Fading (Manual)
K_dB = 15;
K = 10^(K_dB / 10);
s = sqrt(K / (K + 1));
sigma = sqrt(1 / (2 * (K + 1)));
rician_fading = s + sigma * (randn(1, length(sigtx)) + 1j * randn(1, length(sigtx)));
faded_signal = real(rician_fading) .* sigtx;

%% BER Analysis under Rician Fading + Manual AWGN
snr_dBs = 0:1:10;
ber = zeros(size(snr_dBs));

signal_power = mean(faded_signal .^ 2);

for m = 1:length(snr_dBs)
    snr_linear = 10^(snr_dBs(m) / 10);
    noise_power = signal_power / snr_linear;
    noise = sqrt(noise_power) * randn(1, length(faded_signal));
    rx_signal = faded_signal + noise;

    %% Despread
    despread = rx_signal .* pnupsampled;

    %% BPSK Demodulation
    demod_carrier = sqrt(2 * eb) * cos(2 * pi * fc * t);
    demod = despread .* demod_carrier;

    rxbits = [];
    for i = 1:length_user
        bit_chunk = demod((i - 1) * 10 + 1 : i * 10);
        rxbits(i) = sum(bit_chunk) > 0;
    end

    %% BER Calculation (Compare with encoded bits)
    ber(m) = sum(rxbits ~= (user == 1)) / length_user;
end

%% Plot BER
figure(4)
plot(snr_dBs, ber, '-o');
grid on;
xlabel('SNR (dB)');
ylabel('BER');
title('BER vs SNR under AWGN + Rician fading (No Toolbox)');
legend('Simulated BER');
