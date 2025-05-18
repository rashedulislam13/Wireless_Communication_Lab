clear all;
close all;

% Parameters
f = 1000;        % Frequency of the sinusoidal signal (Hz)
Fs = 4000;       % Sampling frequency (samples per second)
t = (1/Fs:1/Fs:1)';  % Time vector for 1 second
Am = 1.0;        % Amplitude of the sinusoid

% Generate signal
signal = Am * sin(2 * pi * f * t);

% Plot original signal
figure(1);
plot(t(1:200), signal(1:200));
set(gca, 'ytick', [-1 0 1]);
title('Segment of the Synthetic Sinusoidal Waveform');
xlabel('Time (sec)');
ylabel('Amplitude (volt)');
grid on;

% Quantization parameters
max_val = max(signal);
min_val = min(signal);
n_bits = 8;      % 8 bits per sample
n_levels = 2^n_bits;  % 256 levels
interval = (max_val - min_val) / (n_levels - 1);

partition = min_val:interval:max_val;
codebook = (min_val - interval/2):interval:(max_val + interval/2); % Adjusted for quantization

% Manual quantization (closest codebook value)
index = zeros(size(signal));
for i = 1:length(signal)
    [~, idx] = min(abs(codebook - signal(i)));
    index(i) = idx - 1;  % Zero-based indexing (0 to 255)
end

% Convert decimal indices to binary (8 bits per sample)
num_samples = length(signal);
matrix = zeros(num_samples, n_bits);
for i = 1:num_samples
    % Convert index to binary, MSB first
    bin_str = dec2bin(index(i), n_bits);
    matrix(i, :) = double(bin_str) - '0';
end

% Prepare baseband bitstream
matrixtps = matrix';  % 8 rows x num_samples columns
baseband = reshape(matrixtps, [], 1);  % Column vector of bits

% Bit rate parameters
Tb = 1 / (Fs * n_bits);  % Bit duration for 8 bits per sample
time_vec = 0:Tb:(length(baseband)-1)*Tb;

% Plot baseband signal
figure(2);
stairs(time_vec(1:500), baseband(1:500));
title('Segment of Baseband Signal');
xlabel('Time (sec)');
ylabel('Binary Value');
set(gca, 'ytick', [0 1]);
axis([0 time_vec(500) -0.1 1.1]);
grid on;

% QPSK Modulation Parameters
M = 4;  % QPSK
k = log2(M);  % Bits per symbol (2 for QPSK)

% Ensure baseband length is divisible by k
len = length(baseband);
if mod(len, k) ~= 0
    baseband = [baseband; zeros(k - mod(len, k), 1)];
end

% Convert bits to symbols (manual bi2de)
num_symbols = floor(length(baseband) / k);
symbols = zeros(num_symbols, 1);
for i = 1:num_symbols
    bit_pair = baseband((i-1)*k + 1:i*k);
    % Map bit pairs to decimal: 00->0, 01->1, 10->2, 11->3
    symbols(i) = bit_pair(1) * 2 + bit_pair(2);
end

% QPSK Modulation (manual pskmod)
% QPSK constellation: 0 -> 1+1j, 1 -> -1+1j, 2 -> 1-1j, 3 -> -1-1j
constellation = [1+1j, -1+1j, 1-1j, -1-1j] / sqrt(2); % Normalized
modulated_signal = zeros(num_symbols, 1);
for i = 1:num_symbols
    modulated_signal(i) = constellation(symbols(i) + 1); % +1 for 1-based indexing
end

% Channel (no AWGN as per original code)

% QPSK Demodulation (manual pskdemod)
demodulated_symbols = zeros(num_symbols, 1);
for i = 1:num_symbols
    % Find closest constellation point
    [~, idx] = min(abs(modulated_signal(i) - constellation));
    demodulated_symbols(i) = idx - 1; % Convert to 0-based (0,1,2,3)
end

% Symbol Error Rate (manual symerr)
symbol_errors = sum(symbols ~= demodulated_symbols);
symbol_error_rate = symbol_errors / num_symbols;

% Convert symbols to bits (manual de2bi)
retrieved_bits = zeros(num_symbols * k, 1);
for i = 1:num_symbols
    % Map symbol to bit pair
    sym = demodulated_symbols(i);
    if sym == 0
        bits = [0; 0];
    elseif sym == 1
        bits = [0; 1];
    elseif sym == 2
        bits = [1; 0];
    else % sym == 3
        bits = [1; 1];
    end
    retrieved_bits((i-1)*k + 1:i*k) = bits;
end

% Bit Error Rate (manual biterr)
bit_errors = sum(baseband(1:length(retrieved_bits)) ~= retrieved_bits);
bit_error_rate = bit_errors / length(retrieved_bits);

% Reshape bits to bytes (8 bits per sample)
decoded_bits_reshaped = reshape(retrieved_bits(1:floor(length(retrieved_bits)/n_bits)*n_bits), n_bits, [])';

% Convert binary to decimal values (manual bi2de)
int_values = zeros(size(decoded_bits_reshaped, 1), 1);
for i = 1:size(decoded_bits_reshaped, 1)
    int_values(i) = bin2dec(char(decoded_bits_reshaped(i, :) + '0'));
end

% Decimal Value Error Rate (manual biterr)
decimal_errors = sum(index(1:length(int_values)) ~= int_values);
decimal_error_rate = decimal_errors / length(int_values);

% Reconstruct signal from quantization
reconstructed_signal = min_val + int_values * interval;

% Plot original vs reconstructed signal
figure(3);
subplot(2,1,1);
plot(t(1:100), signal(1:100));
title('Original Signal Segment');
xlabel('Time (sec)');
ylabel('Amplitude');
set(gca, 'ytick', [-1 0 1]);
axis([0 t(100) -1 1]);
grid on;

subplot(2,1,2);
plot(t(1:min(100, length(reconstructed_signal))), reconstructed_signal(1:min(100, length(reconstructed_signal))));
title('Reconstructed Signal Segment after QPSK Demodulation');
xlabel('Time (sec)');
ylabel('Amplitude');
set(gca, 'ytick', [-1 0 1]);
axis([0 t(100) -1 1]);
grid on;

% Display results
fprintf('Symbol Error Rate (SER): %.5f\n', symbol_error_rate);
fprintf('Bit Error Rate (BER): %.5f\n', bit_error_rate);
fprintf('Decimal Value Error Rate: %.5f\n', decimal_error_rate);