clc;
clear;
close all;

%% ------------------------ Input Data ------------------------
xbit = [1 0 1 1 0 1 0 0 0 1 1 0];  % Original binary bit stream

% Initial reference bit is assumed to be 1
difencod(1) = ~(1 - xbit(1));
for i = 2:length(xbit)
    difencod(i) = ~(difencod(i-1) - xbit(i));
end

% Recover original bits (for verification)
xbit(1) = 1 - ~(difencod(1));
for i = 2:length(xbit)
    xbit(i) = difencod(i-1) - ~(difencod(i));
    if (xbit(i) == -1)
        xbit(i) = 1;
    end
end

%% ------------------------ I & Q Mapping ------------------------
% In-phase (I) unipolar mapping
for i = 1:2:length(difencod)-1
    inp(i) = difencod(i);
    inp(i+1) = inp(i);
end

% Quadrature (Q) unipolar mapping
for i = 2:2:length(difencod)
    qp(i) = difencod(i);
    qp(i-1) = qp(i);
end

% I: Bipolar NRZ mapping
for i = 1:length(inp)
    if (inp(i) == 1)
        it(i) = 1;
    else
        it(i) = -1;
    end
end

% Q: Bipolar NRZ mapping
for i = 1:length(qp)
    if (qp(i) == 1)
        qt(i) = 1;
    else
        qt(i) = -1;
    end
end

%% ------------------------ Raised Cosine Filter ------------------------
filtorder = 40;       % Filter order
nsamp = 4;            % Samples per symbol
rolloff = 0.5;        % Rolloff factor
span = filtorder / nsamp;  % Span in symbols

% Design Root Raised Cosine filter
rrcfilter = rcosdesign(rolloff, span, nsamp, 'normal');

% Plot impulse response
figure;
impz(rrcfilter, 1);
grid on;
title('Impulse Response of Raised Cosine Filter', 'Interpreter', 'none');

%% ------------------------ Transmit Signal ------------------------
% Upsample and filter I and Q components
itx = upfirdn(it, rrcfilter, nsamp, 1);
qtx = upfirdn(qt, rrcfilter, nsamp, 1);

% Time vectors
Drate = 64000;
T = 1/Drate;
Ts = T / nsamp;
time = 0:Ts:(length(itx)-1)*Ts;
tme = 0:Ts:(length(qtx)-1)*Ts;

% Plot I component
figure;
plot(time, itx);
xlabel('Time (sec)');
ylabel('Amplitude (V)');
title('Filtered In-phase Component (I)', 'Interpreter', 'none');
grid on;

% Plot Q component
figure;
plot(tme, qtx);
xlabel('Time (sec)');
ylabel('Amplitude (V)');
title('Filtered Quadrature Component (Q)', 'Interpreter', 'none');
grid on;

% Carrier frequency for OQPSK
fc = 900e6;  % 900 MHz
dd = 2*pi*fc*time';
ddd = 2*pi*fc*tme';

% Apply symbol delay for Q channel (OQPSK offset)
delay = zeros(size(qtx));
delay(nsamp+1:end) = qtx(1:end-nsamp);

% OQPSK Modulated signal
mt = (cos(dd) .* itx') + (sin(ddd) .* delay');

figure;
plot(time, mt);
xlabel('Time (sec)');
ylabel('Amplitude (V)');
title('OQPSK Modulated Signal (Differential Encoding)', 'Interpreter', 'none');
grid on;

%% ------------------------ Add AWGN (Custom) ------------------------
snr = 10;  % in dB
signal_power = mean(abs(mt).^2);
snr_linear = 10^(snr/10);
noise_power = signal_power / snr_linear;
noise = sqrt(noise_power) * randn(size(mt));
madd = mt + noise;

figure;
plot(time, madd);
xlabel('Time (sec)');
ylabel('Amplitude (V)');
title('OQPSK Signal with Added White Noise', 'Interpreter', 'none');
grid on;

%% ------------------------ Demodulation ------------------------
% Multiply by carrier to get baseband
cscomp = madd .* cos(dd);
sincomp = madd .* sin(ddd);

% Low-pass filtering to recover I and Q
lpfin = upfirdn(cscomp, rrcfilter, 1, 1);
lpfqu = upfirdn(sincomp, rrcfilter, 1, 1);

% Time axes
tmx = 0:Ts:(length(lpfin)-1)*Ts;
tmy = 0:Ts:(length(lpfqu)-1)*Ts;

figure;
plot(tmx, lpfin);
xlabel('Time (sec)');
ylabel('Amplitude');
title('Demodulated In-phase Component (I)', 'Interpreter', 'none');
grid on;

figure;
plot(tmy, lpfqu);
xlabel('Time (sec)');
ylabel('Amplitude');
title('Demodulated Quadrature Component (Q)', 'Interpreter', 'none');
grid on;

%% ------------------------ Bit Decision ------------------------
half = filtorder/2;

% Sample recovered signal at symbol instants
itxx = lpfin(half:nsamp:length(xbit)*nsamp+half-1);
ityy = lpfqu(half:nsamp:length(xbit)*nsamp+half-1);

% Hard decision on I & Q
chk1 = sign(itxx);
chk2 = sign(ityy);

% Convert to binary (+1 → 1, -1 → 0)
for i = 1:length(chk1)
    if chk1(i) > 0
        chk1(i) = 1;
    else
        chk1(i) = -1;
    end
end

for i = 1:length(chk2)
    if chk2(i) > 0
        chk2(i) = 1;
    else
        chk2(i) = -1;
    end
end

disp('I channel bit stream distortion (MSE):');
disp(mean((it - chk1).^2));

disp('Q channel bit stream distortion (MSE):');
disp(mean((qt - chk2).^2));

%% ------------------------ Differential Decoding ------------------------
% Combine I & Q bits into differential stream
for i = 1:2:length(xbit)-1
    dfd(i) = chk1(i);
end
for i = 2:2:length(xbit)
    dfd(i) = chk2(i);
end

% Convert to binary
for i = 1:length(xbit)
    if dfd(i) == 1
        dfdecod(i) = 1;
    else
        dfdecod(i) = 0;
    end
end

% Recover original bits from differential decoding
detected(1) = 1 - ~(dfdecod(1));
for i = 2:length(xbit)
    detected(i) = dfdecod(i-1) - ~(dfdecod(i));
    if detected(i) == -1
        detected(i) = 1;
    end
end

%% ------------------------ Results ------------------------
disp('Bit Stream Distortion (Transmitted vs Detected):');
disp(mean((xbit - detected).^2));  % Mean Square Error

tmx = 0:(1/Drate):(1/Drate)*(length(xbit)-1);

figure;
subplot(2,1,1);
stairs(tmx, xbit, 'LineWidth', 1.5);
ylim([-0.2 1.2]);
xlabel('Time (sec)');
ylabel('Bit');
title('Original Transmitted Bit Stream', 'Interpreter', 'none');
grid on;

subplot(2,1,2);
stairs(tmx, detected, 'LineWidth', 1.5);
ylim([-0.2 1.2]);
xlabel('Time (sec)');
ylabel('Bit');
title('Recovered Bit Stream After OQPSK Demodulation', 'Interpreter', 'none');
grid on;
