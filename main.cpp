
#include <cstdlib>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <mpi.h>
#include <omp.h>


using namespace std;
using namespace  cv;


void watershedSegmentation(cv::Mat& inputImage, cv::Mat& outputImage) {
    // Convert the input image to grayscale
    cv::Mat grayImage;
    cv::cvtColor(inputImage, grayImage, cv::COLOR_BGR2GRAY);

    // Apply thresholding to create a binary image
    cv::Mat binaryImage;
    cv::threshold(grayImage, binaryImage, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // Apply distance transform to the binary image
    cv::Mat distanceTransform;
    cv::distanceTransform(binaryImage, distanceTransform, cv::DIST_L2, 5);

    // Localize markers for watershed
    cv::Mat markers;
    cv::connectedComponents(binaryImage, markers);

    // Apply watershed algorithm
    cv::watershed(inputImage, markers);

    // Output the segmented image
    markers.convertTo(outputImage, CV_8U);
}

int main(int argc, char** argv) {
    int rank, size;

    // Initialize MPI
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Load the image on the root process (rank 0)
    string file= "/home/picox/Projects/Netbeans/Parallel_Watershed_Segment/img.png";
    cv::Mat inputImage, outputImage;
    
    if (rank == 0) {
        inputImage = cv::imread(file, cv::IMREAD_COLOR);
        if (inputImage.empty()) {
            std::cerr << "Error: Unable to load the input image." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Broadcast image dimensions to all processes
    int imageRows, imageCols;
    if (rank == 0) {
        imageRows = inputImage.rows;
        imageCols = inputImage.cols;
    }
    MPI_Bcast(&imageRows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&imageCols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate memory for the partial image data on each process
    int partialRows = imageRows / size;
    cv::Mat partialInput(partialRows, imageCols, CV_8UC3);
    cv::Mat partialOutput(partialRows, imageCols, CV_8UC1);

    // Scatter the input image data among processes
    int counts[size], displs[size];
    if (rank == 0) {
        for (int i = 0; i < size; ++i) {
            counts[i] = partialRows * imageCols * 3;
            displs[i] = i * counts[i];
        }
    }
    MPI_Scatterv(inputImage.data, counts, displs, MPI_BYTE, partialInput.data,
                 counts[rank], MPI_BYTE, 0, MPI_COMM_WORLD);

    // Perform the watershed segmentation in parallel using OpenMP
    #pragma omp parallel for
    for (int i = 0; i < partialRows; ++i) {
        cv::Mat partialInputRow = partialInput.row(i);
        cv::Mat partialOutputRow = partialOutput.row(i);
        watershedSegmentation(partialInputRow, partialOutputRow);
    }

    // Gather the segmented data back to the root process
    MPI_Gatherv(partialOutput.data, partialRows * imageCols, MPI_BYTE,
                outputImage.data, counts, displs, MPI_BYTE, 0, MPI_COMM_WORLD);

    // Display or save the output image on the root process
    namedWindow("Segmented Image", WINDOW_KEEPRATIO);
    if (rank == 0) {
        cv::imshow("Segmented Image", outputImage);
        cv::waitKey(0);
        destroyAllWindows();
    }
    

    // Finalize MPI
    MPI_Finalize();
    return 0;
}

