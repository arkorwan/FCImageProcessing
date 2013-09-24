//
//  FCImageConverter.mm
//
//  from opencv tutorial: http://docs.opencv.org/trunk/doc/tutorials/ios/image_manipulation/image_manipulation.html
//

#import <Foundation/Foundation.h>

@interface FCImageConverter : NSObject

+ (cv::Mat)cvMatFromUIImage:(UIImage *)image;

+ (cv::Mat)cvMatGrayFromUIImage:(UIImage *)image;

+ (UIImage *)UIImageFromCVMat:(cv::Mat)cvMat;

@end
