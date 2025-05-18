clear; close all;

%% 1. Generate Sinusoidal Signal
Fs = 4000;         % Sampling frequency
f = 1000;          % Signal frequency
t = 1/Fs:1/Fs:1;   % Time vector
Am = 1.0;
signal = Am * sin(2*pi*f*t);

figure(1);
plot(t(1:200), signal(1:200));
title('Original Sinusoidal Signal'); xlabel('Time (s)'); ylabel('Amplitude'); grid on;

%% 2. Uniform Quantization (Manual)
L = 256;  % 8-bit
min_val = min(signal);
max_val = max(signal);
q_step = (max_val - min_val) / (L - 1);
codebook = linspace(min_val, max_val, L);
index = zeros(length(signal), 1);

for i = 1:length(signal)
    [~, idx] = min(abs(signal(i) - codebook));
    index(i) = idx - 1;  % 0-based
end

%% 3. Custom Decimal to Binary (No de2bi)
bin_matrix = zeros(length(index), 8);
for i = 1:length(index)
    bin_matrix(i, :) = dec2bin_vec(index(i), 8);
end

baseband = reshape(bin_matrix', [], 1);  % 32000 x 1

%% 4. Show Baseband
Tb = 1/32000;
time_bb = 0:Tb:(length(baseband)-1)*Tb;
figure(2);
stairs(time_bb(1:500), baseband(1:500));
title('Baseband Signal'); xlabel('Time (s)'); ylabel('Bit'); grid on;

%% 5. Simple Convolutional Encoding (Rate 1/2, Constraint Length 7)
G1 = [1 1 1 1 0 0 1]; % 171
G2 = [1 0 1 1 0 1 1]; % 133
K = 7;
state = zeros(1, K-1);
encoded = [];

for i = 1:length(baseband)
    input_bit = baseband(i);
    reg = [input_bit state];
    out1 = mod(sum(reg .* G1), 2);
    out2 = mod(sum(reg .* G2), 2);
    encoded = [encoded out1 out2];
    state = reg(1:end-1);
end

%% 6. Interleaving using Random Permutation
rng(4831);
perm = randperm(length(encoded));
interleaved = encoded(perm);

%% 7. BPSK Modulation
bpsk_signal = 2 * interleaved - 1;

%% (Optional) Add AWGN
% bpsk_signal = bpsk_signal + 0.1 * randn(size(bpsk_signal));

%% 8. Demodulation
demod = double(bpsk_signal > 0);

%% 9. Deinterleaving
[~, revperm] = sort(perm);
deinterleaved = demod(revperm);

%% 10. Simple Hard Decoding (Placeholder)
decoded = zeros(1, floor(length(deinterleaved)/2));
for i = 1:length(decoded)
    bits = deinterleaved((i-1)*2 + (1:2));
    decoded(i) = xor(bits(1), bits(2));  % Dummy decoder
end

%% 11. Reshape Decoded Bits into Bytes
decoded = decoded(:);
decoded = decoded(1:floor(length(decoded)/8)*8);
bit_matrix = reshape(decoded, 8, [])';

%% 12. Custom Binary to Decimal (No bi2de)
decimal_values = zeros(size(bit_matrix, 1), 1);
for i = 1:size(bit_matrix, 1)
    decimal_values(i) = bin2dec_vec(bit_matrix(i, :));
end

%% 13. Reconstruct Signal
reconstructed_signal = codebook(decimal_values + 1);

%% 14. Plot Comparison
figure(3);
subplot(2,1,1);
plot(t(1:100), signal(1:100));
title('Original Signal'); ylabel('Amplitude'); grid on;

subplot(2,1,2);
plot(t(1:100), reconstructed_signal(1:100));
title('Reconstructed Signal'); xlabel('Time (s)'); ylabel('Amplitude'); grid on;

%% --- Helper Functions (No Toolbox Needed) ---

function bin = dec2bin_vec(num, bits)
    bin = zeros(1, bits);
    for i = bits:-1:1
        bin(i) = mod(num, 2);
        num = floor(num / 2);
    end
end

function dec = bin2dec_vec(bin)
    bits = length(bin);
    dec = 0;
    for i = 1:bits
        dec = dec + bin(i) * 2^(bits - i);
    end
end
