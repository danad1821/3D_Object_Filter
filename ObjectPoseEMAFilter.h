#ifndef OBJECT_POSE_EMA_FILTER_H
#define OBJECT_POSE_EMA_FILTER_H

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <pangolin/pangolin.h>

namespace ORB_SLAM3 {
// Define a struct for smoothed pose state
struct SmoothedPoseState {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW 
  Eigen::Vector3d translation;
  Eigen::Quaterniond rotation;
  double lastUpdateTimestamp; 
};

class ObjectPoseEMAFilter {
public:
  ObjectPoseEMAFilter(const std::string &strSettingPath = "");

  // Configuration method
  bool parseFilterParamFile(cv::FileStorage &fSettings);

  // Main filtering function
  Eigen::Matrix4f filterPose(const Eigen::Matrix4f &currentRawPose, float scale);

private:
  // Internal state variables for the filter
  SmoothedPoseState mCurrentSmoothedPose; // First pass smoothed pose
  SmoothedPoseState mPreviousSmoothedPoseForSecondPass; // Second pass smoothed pose 

  bool mIsFirstPose; // Flag for initial state
  
  // the previous dt value for comparison
  double mPreviousDtValue = 0.0; 
  // The calculated EMA smoothing factor (alpha)
  float mCurrentSmoothingFactorAlpha = 0.0f;

  // Configuration parameters (set from YAML)
  float mSmoothingTimeConstant;
  float mDtChangeSignificanceFactor;
  float mMaxNumberOfObjectsToFilter;

  // Filter state (default is active)
  bool mIsFilterActive; // Flag to enable/disable the filter

  // Helper functions for pose decomposition/composition
  Eigen::Vector3d getTranslation(const Eigen::Matrix4f &matrix);
  Eigen::Quaterniond rotationMatrixToQuaternion(const Eigen::Matrix3d &R);
  Eigen::Vector3d getEulerAngles(const Eigen::Matrix3d &mr);

  Eigen::Matrix3d extractRotationWithoutScale(const Eigen::Matrix4d &poseMatrix,
                                              const double scale);
  Eigen::Matrix4f constructPoseMatrix(const Eigen::Vector3d &translationValues,
                                         const Eigen::Quaterniond &quaternionValues,
                                         const Eigen::Matrix3d &scaling);

  // Helper for dt-based alpha calculation
  bool isTheChangeInDtSignificant(double dt);
};

} // namespace ORB_SLAM3

#endif // OBJECT_POSE_EMA_FILTER_H