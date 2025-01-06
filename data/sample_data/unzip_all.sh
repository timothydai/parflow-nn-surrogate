mkdir data/sample_data/stage1
mkdir data/sample_data/stage2
mkdir data/sample_data/stage3
mkdir data/sample_data/autoencoder

# Create 4 copies of the same simulation to create adequately sized training sets.
for i in {0..3}; do
    unzip data/sample_data/stage1.zip -d "data/sample_data/stage1/sim$i"
    unzip data/sample_data/stage2.zip -d "data/sample_data/stage2/sim$i"
    unzip data/sample_data/autoencoder -d "data/sample_data/autoencoder/sim$i"
done

# For Stage 3 simulations, create two copies each of monthly flooding and one-time flooding.
for i in {0..1}; do
    unzip data/sample_data/stage3_monthly.zip -d "data/sample_data/stage3/monthly$i"
    unzip data/sample_data/stage3_once.zip -d "data/sample_data/stage3/once$i"
done

