//
//  FCTextBinarizer.h
//
//  Created by Worakarn Isaratham on 9/23/13.
//

#import <Foundation/Foundation.h>

@interface FCTextBinarizer : NSObject

@property (nonatomic) float maxBoxAspectRatio; // >= 1
@property (nonatomic) NSUInteger minSiblings;
@property (nonatomic) NSUInteger maxChildren;
@property (nonatomic) float maxBoxHeightRatio; //0>=x>=1
@property (nonatomic) float maxBoxWidthRatio; //0>=x>=1


+ (id) sharedInstance;
-(UIImage *) binarizeUIImage:(UIImage *) source;

@end
