wget "https://drive.google.com/uc?export=download&id=15sGhe1qTjyKaUU-DJ5MJ1C-tBecZrjsY" -O dataset/trainning/samples.npy

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=160x8s90jSv2lYLzzvGNKlgQzL9QIhz6t' -O- | gsed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=160x8s90jSv2lYLzzvGNKlgQzL9QIhz6t" -O dataset/trainning/targets.npy && rm -rf /tmp/cookies.txt
