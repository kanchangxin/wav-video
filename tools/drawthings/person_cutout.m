#import <CoreImage/CoreImage.h>
#import <Foundation/Foundation.h>
#import <ImageIO/ImageIO.h>
#import <Vision/Vision.h>

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    if (argc != 3) {
      fprintf(stderr, "Usage: person_cutout input.png output.png\n");
      return 2;
    }

    NSURL *input = [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]];
    NSURL *output = [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[2]]];
    CGImageSourceRef source = CGImageSourceCreateWithURL((__bridge CFURLRef)input, NULL);
    if (!source) return 1;
    CGImageRef image = CGImageSourceCreateImageAtIndex(source, 0, NULL);
    CFRelease(source);
    if (!image) return 1;

    VNGeneratePersonSegmentationRequest *request = [[VNGeneratePersonSegmentationRequest alloc] init];
    request.qualityLevel = VNGeneratePersonSegmentationRequestQualityLevelAccurate;
    request.outputPixelFormat = kCVPixelFormatType_OneComponent8;
    VNImageRequestHandler *handler = [[VNImageRequestHandler alloc] initWithCGImage:image options:@{}];
    NSError *error = nil;
    BOOL ok = [handler performRequests:@[request] error:&error];
    if (!ok || request.results.count == 0) {
      fprintf(stderr, "Person segmentation failed: %s\n", error.localizedDescription.UTF8String);
      CGImageRelease(image);
      return 1;
    }

    VNPixelBufferObservation *observation = request.results.firstObject;
    CIImage *original = [CIImage imageWithCGImage:image];
    CIImage *mask = [CIImage imageWithCVPixelBuffer:observation.pixelBuffer];
    CGFloat scaleX = original.extent.size.width / mask.extent.size.width;
    CGFloat scaleY = original.extent.size.height / mask.extent.size.height;
    mask = [mask imageByApplyingTransform:CGAffineTransformMakeScale(scaleX, scaleY)];
    CIImage *clear = [[CIImage imageWithColor:[CIColor colorWithRed:0 green:0 blue:0 alpha:0]]
        imageByCroppingToRect:original.extent];
    CIFilter *blend = [CIFilter filterWithName:@"CIBlendWithMask"];
    [blend setValue:original forKey:kCIInputImageKey];
    [blend setValue:clear forKey:kCIInputBackgroundImageKey];
    [blend setValue:mask forKey:kCIInputMaskImageKey];
    CIImage *result = blend.outputImage;
    CIContext *context = [CIContext context];
    CGImageRef rendered = [context createCGImage:result fromRect:original.extent];
    CGImageRelease(image);
    if (!rendered) return 1;

    CGImageDestinationRef destination = CGImageDestinationCreateWithURL(
        (__bridge CFURLRef)output, CFSTR("public.png"), 1, NULL);
    if (!destination) {
      CGImageRelease(rendered);
      return 1;
    }
    CGImageDestinationAddImage(destination, rendered, NULL);
    BOOL saved = CGImageDestinationFinalize(destination);
    CFRelease(destination);
    CGImageRelease(rendered);
    if (!saved) return 1;
    printf("Wrote: %s\n", argv[2]);
    return 0;
  }
}
