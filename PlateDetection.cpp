#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cmath>
#include <algorithm> 


using namespace cv;
using namespace std;

const double PERCENT_DISTANCE_CONTOUR = 0.02; // Epsilon in approxPolyDP
const double MIN_AREA_CIRCLE = 500;           // Minimum area for a plate
const int NUM_POINTS_CIRCLE = 8;              // Number of points to approximate a circle
const double ASPECT_RATIO_THRESHOLD = 0.75;
const double CLEANLINESS_THRESHOLD = 10;


void seeProcess(const Mat& image, const string& windowName)
{
    namedWindow(windowName, WINDOW_AUTOSIZE);
    imshow(windowName, image);
    waitKey(0);
    destroyWindow(windowName);
}

void createColorHistogram(const Mat& plateImage, Mat& hist, int& bucketSize) {
    const int size = 4;
    bucketSize = 256 / size;
    int dims[] = { size, size, size };

    hist = Mat(3, dims, CV_32S, Scalar::all(0));

    for (int r = 0; r < plateImage.rows; r++) {
        for (int c = 0; c < plateImage.cols; c++) {
            Vec3b color = plateImage.at<Vec3b>(r, c);
            int b = color[0] / bucketSize;
            int g = color[1] / bucketSize;
            int red = color[2] / bucketSize;

            hist.at<int>(red, g, b) += 1;
        }
    }
}


Vec3i findMostCommonColor(const Mat& hist, int bucketSize) {
    int maxVal = 0;
    Vec3i mostCommonColor = Vec3i(0, 0, 0);

    for (int r = 0; r < hist.size[0]; r++) {
        for (int g = 0; g < hist.size[1]; g++) {
            for (int b = 0; b < hist.size[2]; b++) {
                int val = hist.at<int>(r, g, b);
                if (val > maxVal) {
                    maxVal = val;
                    mostCommonColor = Vec3i(r, g, b);
                }
            }
        }
    }

    return mostCommonColor;
}

bool isPatternPlate1(const Mat& plateImage) {
    Mat hist;
    int bucketSize;
    createColorHistogram(plateImage, hist, bucketSize);

    Vec3i mostCommonColor = findMostCommonColor(hist, bucketSize);

    int maxVal = hist.at<int>(mostCommonColor[0], mostCommonColor[1], mostCommonColor[2]);
    const int patternThreshold = 0.5 * plateImage.rows * plateImage.cols;

    return maxVal < patternThreshold;
}

bool isWhitePlate(const Mat& plateImage) {
    // Define the threshold for white 
    const Scalar lowerWhiteThresh(200, 200, 200); // Slightly white
    const Scalar upperWhiteThresh(255, 255, 255); // Pure white 

    // Count the number of white (or slightly white) pixels
    int whitePixels = 0;

    // Loop over all pixels and count how many are within the white range
    for (int y = 0; y < plateImage.rows; ++y)
    {
        for (int x = 0; x < plateImage.cols; ++x)
        {
            Vec3b pixel = plateImage.at<Vec3b>(y, x);
            if (pixel[0] >= lowerWhiteThresh[0] && pixel[1] >= lowerWhiteThresh[1] && pixel[2] >= lowerWhiteThresh[2] &&
                pixel[0] <= upperWhiteThresh[0] && pixel[1] <= upperWhiteThresh[1] && pixel[2] <= upperWhiteThresh[2])
            {
                whitePixels++;
            }
        }
    }

    // Calculate the percentage of white pixels
    double whitePercentage = static_cast<double>(whitePixels) / (plateImage.rows * plateImage.cols);
    // Threshold 70%
    const double whiteThreshold = 0.7;

    return whitePercentage >= whiteThreshold;
}

bool isWhitePlate1(const Mat& plateImage) {
    Mat hist;
    int bucketSize;
    createColorHistogram(plateImage, hist, bucketSize);

    Vec3i mostCommonColor = findMostCommonColor(hist, bucketSize);

    // Convert the most common color from histogram indices to actual RGB values
    int cRed = mostCommonColor[0] * bucketSize + bucketSize / 2;
    int cGreen = mostCommonColor[1] * bucketSize + bucketSize / 2;
    int cBlue = mostCommonColor[2] * bucketSize + bucketSize / 2;

    // Define thresholds for white color
    const int lowerThreshold = 200; // Lower bound to consider a color as white
    const int upperThreshold = 255; // Upper bound (max value for a color component)

    // Check if the most common color is within the range for white
    return (cRed >= lowerThreshold && cRed <= upperThreshold) &&
        (cGreen >= lowerThreshold && cGreen <= upperThreshold) &&
        (cBlue >= lowerThreshold && cBlue <= upperThreshold);
}

vector<Rect> findPlates(Mat& image, vector<vector<Point>>& plateContours)
{
    vector<Rect> plates;
    Mat grayImage;
    cvtColor(image, grayImage, COLOR_BGR2GRAY);
    //seeProcess(grayImage, "Grayscale Image");
    GaussianBlur(grayImage, grayImage, Size(3, 3), 2.5, 2.5);

    int lowerThreshold = 50;
    int upperThreshold = 150;
    Canny(grayImage, grayImage, lowerThreshold, upperThreshold);

    Mat element = getStructuringElement(MORPH_RECT, Size(4.5, 4.5), Point(1, 1));
    dilate(grayImage, grayImage, element);

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(grayImage, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    for (auto& contour : contours)
    {
        double area = contourArea(contour);

        if (area >= MIN_AREA_CIRCLE)
        {
            double epsilon = PERCENT_DISTANCE_CONTOUR * arcLength(contour, true);
            vector<Point> approxCurve;
            approxPolyDP(contour, approxCurve, epsilon, true);

            if (approxCurve.size() >= NUM_POINTS_CIRCLE)
            {
                Rect boundingRect = cv::boundingRect(contour);
                double aspectRatio = static_cast<double>(boundingRect.width) / boundingRect.height;

                if (aspectRatio >= ASPECT_RATIO_THRESHOLD && aspectRatio <= 1 / ASPECT_RATIO_THRESHOLD)
                {
                    plates.push_back(boundingRect);
                    plateContours.push_back(contour);
                }
            }
        }
    }

    return plates;
}


bool isPatternPlate(const Mat& plateImage) {
    Ptr<SIFT> detector = SIFT::create();
    vector<KeyPoint> keypoints;
    // Initialize SIFT detector
    Mat descriptors;

    // Detect SIFT features in the plate image
    detector->detectAndCompute(plateImage, noArray(), keypoints, descriptors);

    if (keypoints.empty()) return false;

    // Number of clusters
    int k = 5;
    Mat labels;
    vector<Point2f> points;
    for (const auto& kp : keypoints)
    {
        points.push_back(kp.pt);
    }
    Mat pointsMat(points.size(), 1, CV_32FC2, &points[0]);
    // Cluster the keypoints using k-means
    kmeans(pointsMat, k, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0), 3, KMEANS_PP_CENTERS);


    // Calculate the size of each cluster
    vector<int> clusterSizes(k, 0);
    for (int i = 0; i < points.size(); i++)
    {
        int clusterIdx = labels.at<int>(i);
        clusterSizes[clusterIdx]++;
    }

    // Check if clusters are evenly distributed and similar in size
    double avgSize = 0.0;
    for (const auto& size : clusterSizes)
    {
        avgSize += size;
    }
    avgSize /= k;

    bool sizeRegular = true;
    for (const auto& size : clusterSizes)
    {
        if (abs(size - avgSize) > avgSize * 0.5) { // 50% deviation threshold
            sizeRegular = false;
            break;
        }
    }

    return sizeRegular;
}

bool isPlateClean(const Mat& image, const vector<Point>& plateContour)
{
    // Convert the plate region to HSV color space
    Mat hsvImage;
    cvtColor(image, hsvImage, COLOR_BGR2HSV);

    // Create a mask for the plate region
    Mat mask = Mat::zeros(image.size(), CV_8U);
    drawContours(mask, vector<vector<Point>>{plateContour}, 0, Scalar(255), FILLED);

    // Extract the plate region using the mask
    Mat plateRegion;
    image.copyTo(plateRegion, mask);

    // Calculate the mean color intensity of the plate region
    Scalar meanColor = mean(plateRegion);
    // Check if the mean color intensity is below the cleanliness threshold
    return (meanColor[0] < CLEANLINESS_THRESHOLD && meanColor[1] < CLEANLINESS_THRESHOLD && meanColor[2] < CLEANLINESS_THRESHOLD);
}

bool isPatternPlateClean(const Mat& image, const vector<Point>& plateContour)
{
    // Convert the plate region to grayscale
    Mat plateGray;
    cvtColor(image, plateGray, COLOR_BGR2GRAY);

    // Threshold the grayscale image to create a binary mask
    // Adjust the threshold value based on your observations
    int thresholdValue = 120;
    Mat plateMask;
    threshold(plateGray, plateMask, thresholdValue, 255, THRESH_BINARY);

    // Create a mask for the plate region
    Mat mask = Mat::zeros(image.size(), CV_8U);
    drawContours(mask, vector<vector<Point>>{plateContour}, 0, Scalar(255), FILLED);

    // Apply the binary mask to focus on the plate region
    Mat plateRegion;
    image.copyTo(plateRegion, mask);

    // Apply the binary mask to the grayscale image
    Mat plateGrayMasked;
    plateGray.copyTo(plateGrayMasked, plateMask);

    // Calculate the mean intensity of the plate region
    Scalar meanIntensity = mean(plateGrayMasked);

    // Define a threshold for cleanliness (you may need to adjust this value)
    double cleanlinessThreshold = 160;

    // Check if the mean intensity is below the cleanliness threshold
    return meanIntensity[0] > cleanlinessThreshold;
}

int main() {
    Mat image = imread("test11.jpg");
    if (image.empty()) {
        cout << "Could not read the image." << endl;
        return 1;
    }

    vector<vector<Point>> plateContours;
    vector<Rect> plates = findPlates(image, plateContours); // Assume this function is defined elsewhere

    // Draw rectangles around detected plates and label them
    for (size_t i = 0; i < plates.size(); ++i) {
        // Extract the plate image using the bounding rectangle
        Mat plateImage = image(plates[i]);
        String plateLabel;
        String cleanlinessLabel;

        if (isPatternPlate(plateImage)) {
            plateLabel = "Pattern Plate ";
            if (isPatternPlateClean(image, plateContours[i])) {
                cleanlinessLabel = "Clean ";
            }
            else {
                cleanlinessLabel = "Dirty ";
            }
        }
        else if (isWhitePlate(plateImage)) {
            plateLabel = "White Plate ";
            cleanlinessLabel = "Clean ";
        }
        else {
            plateLabel = "White Plate ";
            cleanlinessLabel = "Dirty ";
        }

        // Draw rectangle around the plate
        rectangle(image, plates[i], Scalar(0, 255, 0), 2);

        // Define text properties
        int fontFace = FONT_HERSHEY_SIMPLEX;
        double fontScale = 0.5;
        int thickness = 1;

        // Calculate text size to position it centered over the plate
        int baseline = 0;
        Size textSize = getTextSize(plateLabel, fontFace, fontScale, thickness, &baseline);
        Point textOriginLabel(plates[i].x + (plates[i].width - textSize.width) / 2,
            plates[i].y - (textSize.height + 5));

        // Put the plaet label on the image above the rectangle
        putText(image, plateLabel, textOriginLabel, fontFace, fontScale, Scalar(255, 0, 0), thickness);

        // Calculate text size to position it centered under the plate
        Size textSize1 = getTextSize(cleanlinessLabel, fontFace, fontScale, thickness, &baseline);
        Point textOriginClean(plates[i].x + (plates[i].width - textSize1.width) / 2, plates[i].y + plates[i].height + textSize1.height + 5);

        // Put the cleanliness label on the image above the rectangle
        putText(image, cleanlinessLabel, textOriginClean, fontFace, fontScale, Scalar(255, 0, 0), thickness);
    }

    // Display the result
    namedWindow("Detected Plates", WINDOW_AUTOSIZE);
    imshow("Detected Plates", image);
    waitKey(0);

    return 0;
}
