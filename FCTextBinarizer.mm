//
//  FCTextBinarizer.mm
//
//  Created by Worakarn Isaratham on 9/23/13.
//

#import "FCTextBinarizer.h"
#import <unordered_map>
#import "FCImageConverter.h"

@implementation FCTextBinarizer{
    
    cv::Mat *imageRef;
    NSInteger imageWidth;
    NSInteger imageHeight;
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchies;
    std::vector<cv::Rect> contourRects;
    std::vector<BOOL> isPotentialTextBoxes;
    std::unordered_map<NSUInteger, NSUInteger> childrenCounts;
    std::unordered_map<NSUInteger, NSUInteger> siblingCounts;
}

static FCTextBinarizer *instance;

+ (void)initialize
{
    static BOOL initialized = NO;
    if(!initialized)
    {
        initialized = YES;
        instance = [FCTextBinarizer new];
    }
}

+ (id) sharedInstance
{
    return instance;
}

-(id) init
{
    if(self = [super init]){
        _maxBoxAspectRatio = 20.0;
        _minSiblings = 5;
        _maxChildren = 4;
        _maxBoxHeightRatio = 0.2;
        _maxBoxWidthRatio = 0.2;
    }
    return self;
}

-(BOOL) isBoxProperlyShaped:(NSUInteger) idx
{
    cv::Rect rect = contourRects[idx];
    float w = rect.width * 1.0;
    float h = rect.height * 1.0;
    float r = h > 0 ? w/h : 0;
    if(r < 1/_maxBoxAspectRatio || r > _maxBoxAspectRatio){
        return NO;
    }
    if(w > imageWidth * _maxBoxWidthRatio){
        return NO;
    }
    if(h > imageHeight * _maxBoxHeightRatio){
        return NO;
    }
    return YES;
}

-(BOOL) isContourConnected:(NSUInteger) idx
{
    cv::Point first = contours[idx][0];
    cv::Point last = contours[idx][contours[idx].size() -1];
    return abs(first.x - last.x) <= 1 && abs(first.y - last.y) <= 1;
    
}

-(BOOL) isPotentialTextBox:(NSUInteger) idx
{
    return [self isBoxProperlyShaped:idx] && [self isContourConnected:idx];
}

-(NSInteger) getParentIndex:(NSUInteger) idx
{
    NSInteger parent = hierarchies[idx][3];
    while(parent >= 0 && !isPotentialTextBoxes[parent]){
        parent = hierarchies[parent][3];
    }
    return parent;
}

-(NSUInteger) countSiblings:(NSUInteger) idx
{
    if(siblingCounts.find(idx) != siblingCounts.end()){
        return siblingCounts[idx];
    } else {
        NSUInteger count = [self countChildren:idx];
        
        NSInteger prev = hierarchies[idx][0];
        while(prev >= 0){
            if(isPotentialTextBoxes[prev]){
                count++;
            }
            count+=[self countChildren:prev];
            prev = hierarchies[prev][0];
        }
        
        NSInteger next = hierarchies[idx][1];
        while(next >= 0){
            if(isPotentialTextBoxes[next]){
                count++;
            }
            count+=[self countChildren:next];
            next = hierarchies[next][1];
        }
        siblingCounts.emplace(idx, count);
        return count;
    }
}

-(NSUInteger) countChildren:(NSUInteger) idx
{
    if(childrenCounts.find(idx) != childrenCounts.end()){
        return childrenCounts[idx];
    } else {
        NSUInteger count = 0;
        if(hierarchies[idx][2] >= 0){
            if(isPotentialTextBoxes[hierarchies[idx][2]]){
                count++;
            }
            count += [self countSiblings:hierarchies[idx][2]];
        }
        childrenCounts.emplace(idx, count);
        return count;
    }
}

-(BOOL) hasProperNumberOfSiblingsAndChildren:(NSInteger) idx
{
    NSInteger parent = [self getParentIndex:idx];
    if(parent >= 0 && [self countChildren:parent] < _minSiblings) {
        return NO;
    }
    
    if([self countChildren:idx] > _maxChildren){
        return NO;
    }
    
    return YES;
}

-(float) getIntensityAtPoint:(NSInteger) x :(NSInteger) y
{
    if(x >= imageWidth || y >= imageHeight){
        return 0.0f;
    } else {
        cv::Mat mat = (*imageRef);
        cv::Vec3b pixel = mat.at<cv::Vec3b>(y,x);
        return 0.3 * pixel[2] + 0.59 * pixel[1] + 0.11 * pixel[0];
    }
    
}

-(cv::Mat) mergeChannels:(cv::Mat) src withBlock:(cv::Mat(^)(cv::Mat)) block
{
    std::vector<cv::Mat> channels;
    cv::split(src, channels);
    cv::Mat merge;
    for(int channel = 0; channel < channels.size(); channel++){
        if(channel == 0){
            merge = block(channels[channel]);
        } else {
            cv::Mat result = block(channels[channel]);
            cv::bitwise_or(merge, result, merge);
        }
    }
    return merge;
}

-(cv::Mat) binarize:(cv::Mat) original
{
    
    cv::Mat temp1;
    cv::Mat temp2;
    imageWidth = original.cols;
    imageHeight = original.rows;
    
    //convert to 3-channel image
    cv::cvtColor(original , original , CV_BGRA2BGR);
    cv::bilateralFilter(original, temp1, 5, 100, 100);
    
    imageRef = &temp1;
    
    temp2 = [self mergeChannels:temp1 withBlock:^(cv::Mat channel){
        cv::Mat result;
        cv::Canny(channel, result, 100, 255);
        return result;
    }];
    
    cv::findContours(temp2, contours, hierarchies, CV_RETR_TREE, CV_CHAIN_APPROX_NONE );
    
    std::vector<NSInteger> keepers;
    
    for(int i = 0; i < contours.size(); i++ ) {
        contourRects.push_back(cv::boundingRect(contours[i]));
        isPotentialTextBoxes.push_back([self isPotentialTextBox:i]);
    }
    
    for(int i = 0; i < contours.size(); i++ ) {
        if(isPotentialTextBoxes[i] && [self hasProperNumberOfSiblingsAndChildren:i]){
            keepers.push_back(i);
        }
    }
    
    cv::Mat newImage = cv::Mat(imageHeight, imageWidth, temp2.type());
    newImage.setTo(cv::Scalar(255, 255, 255));
    
    for(int i = 0; i < keepers.size(); i++ ) {
        NSInteger idx = keepers[i];
        std::vector<cv::Point> cnt = contours[idx];
        cv::Rect box = contourRects[idx];
        float fg_int = 0.0;
        for(int j = 0; j < cnt.size(); j++ ) {
            cv::Point p = cnt[j];
            fg_int += [self getIntensityAtPoint:p.x :p.y];
        }
        fg_int /= cnt.size();
        
        
        std::vector<NSInteger> bg_int_samples;
        bg_int_samples.push_back([self getIntensityAtPoint:box.x - 1 :box.y - 1]);
        bg_int_samples.push_back([self getIntensityAtPoint:box.x - 1 :box.y]);
        bg_int_samples.push_back([self getIntensityAtPoint:box.x :box.y - 1]);
        bg_int_samples.push_back([self getIntensityAtPoint:box.x + box.width + 1 :box.y - 1]);
        bg_int_samples.push_back([self getIntensityAtPoint:box.x + box.width :box.y - 1]);
        bg_int_samples.push_back([self getIntensityAtPoint:box.x + box.width + 1 :box.y]);
        bg_int_samples.push_back([self getIntensityAtPoint:box.x - 1 :box.y + box.height + 1]);
        bg_int_samples.push_back([self getIntensityAtPoint:box.x - 1 :box.y + box.height]);
        bg_int_samples.push_back([self getIntensityAtPoint:box.x :box.y +box.height + 1]);
        bg_int_samples.push_back([self getIntensityAtPoint:box.x + box.width + 1 :box.y + box.height + 1]);
        bg_int_samples.push_back([self getIntensityAtPoint:box.x + box.width :box.y + box.height + 1]);
        bg_int_samples.push_back([self getIntensityAtPoint:box.x + box.width + 1 :box.y + box.height]);
        
        //find median
        std::sort(bg_int_samples.begin(), bg_int_samples.end());
        float bg_int = (bg_int_samples[(bg_int_samples.size() - 1) / 2] + bg_int_samples[bg_int_samples.size() / 2]) / 2;
        
        int fg = 255;
        int bg = 0;
        
        if(fg_int < bg_int){
            fg = 0;
            bg = 255;
        }
        
        for(int x = box.x; x<box.x + box.width; x++){
            for(int y = box.y; y < box.y + box.height; y++){
                if(y >= imageHeight || x >= imageWidth){
                    continue;
                }
                newImage.row(y).col(x) = [self getIntensityAtPoint:x :y] > fg_int ? bg : fg;
            }
        }
        
    }
    
    cv::GaussianBlur(newImage, newImage, cv::Size(5,5), 0,0);
    
    imageRef = nil;
    contours.clear();
    hierarchies.clear();
    contourRects.clear();
    isPotentialTextBoxes.clear();
    childrenCounts.clear();
    siblingCounts.clear();
    
    return newImage;
}

-(UIImage *) binarizeUIImage:(UIImage *) source
{
    return [FCImageConverter UIImageFromCVMat:[self binarize:[FCImageConverter cvMatFromUIImage: source]]];
}

@end
