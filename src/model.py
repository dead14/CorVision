import segmentation_models_pytorch as smp
import torch

def create_model():
    """
    Creates a U-Net model with a ResNet34 encoder.
    The encoder is pre-trained on ImageNet, which makes it learn much faster.
    """
    model = smp.Unet(
        encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1,                      # model output channels (number of classes in your dataset)
    )
    return model

if __name__ == "__main__":
    model = create_model()
    # Test with dummy input
    dummy_input = torch.randn(1, 3, 256, 256)
    output = model(dummy_input)
    print("Model output shape:", output.shape) # Should be [1, 1, 256, 256]
