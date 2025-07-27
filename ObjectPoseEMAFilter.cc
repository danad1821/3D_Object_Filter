#include "ObjectPoseEMAFilter.h" // Your header
#include <Eigen/Geometry>        // For Eigen::Quaterniond
#include <algorithm>             // For std::min, std::max
#include <chrono>                // For current time
#include <cmath>                 // For std::exp
#include <iostream>              // For debug/error output
#include <opencv2/opencv.hpp>    // For cv::FileStorage and cv::FileNode

namespace ORB_SLAM3 {

// Constructor: Initialize all member variables
ORB_SLAM3::ObjectPoseEMAFilter::ObjectPoseEMAFilter(
    const std::string &strSettingPath)
    : mIsFirstPose(true), mPreviousDtValue(0.0), mSmoothingTimeConstant(0.0),
      mDtChangeSignificanceFactor(1.2), mMaxNumberOfObjectsToFilter(5),
      mIsFilterActive(true), mCurrentSmoothingFactorAlpha(0.0) {

  // set default for first pass
  mCurrentSmoothedPose.translation.setZero();
  mCurrentSmoothedPose.rotation.setIdentity();
  mCurrentSmoothedPose.lastUpdateTimestamp = 0.0;

  // set default for second pass
  mPreviousSmoothedPoseForSecondPass.translation.setZero();
  mPreviousSmoothedPoseForSecondPass.rotation.setIdentity();
  mPreviousSmoothedPoseForSecondPass.lastUpdateTimestamp = 0.0;

  if (!strSettingPath.empty()) {
    // get setting variables from yaml file
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    if (!fSettings.isOpened()) {
      std::cerr << "Failed to open settings file at: " << strSettingPath
                << std::endl;
    }
    bool is_correct = parseFilterParamFile(fSettings);

    if (!is_correct) {
      std::cerr << "**ERROR in the config file, the format is not correct**"
                << std::endl;
    }
  } else {
    std::cout << "Settings file not provided, using default parameters."
              << std::endl;
  }
}

// Gets the parameters set int the yaml file
bool ORB_SLAM3::ObjectPoseEMAFilter::parseFilterParamFile(
    cv::FileStorage &fSettings) {
  bool b_miss_params = false;

  cv::FileNode node =
      fSettings["ObjectPoseEMAFilter.MaxNumberOfObjectsToFilter"];
  if (!node.empty()) {
    this->mMaxNumberOfObjectsToFilter = node.real();
  } else {
    std::cerr << "*ObjectPoseEMAFilter.MaxNumberOfObjectsToFilter parameter "
                 "doesn't exist*"
              << std::endl;
    b_miss_params = true;
  }

  node = fSettings["ObjectPoseEMAFilter.DtChangeSignificanceFactor"];
  if (!node.empty()) {
    this->mDtChangeSignificanceFactor = node.real();
  } else {
    std::cerr << "*ObjectPoseEMAFilter.DtChangeSignificanceFactor parameter "
                 "doesn't exist*"
              << std::endl;
    b_miss_params = true;
  }

  node = fSettings["ObjectPoseEMAFilter.SmoothingTimeConstant"];
  if (!node.empty()) {
    this->mSmoothingTimeConstant = node.real();
  } else {
    std::cerr
        << "*ObjectPoseEMAFilter.SmoothingTimeConstant parameter doesn't exist*"
        << std::endl;
    b_miss_params = true;
  }

  return !b_miss_params;
}

// Gets translation vector to be smoothed
Eigen::Vector3d
ORB_SLAM3::ObjectPoseEMAFilter::getTranslation(const Eigen::Matrix4f &matrix) {
  Eigen::Map<const Eigen::Matrix<float, 4, 4, Eigen::ColMajor>> eigen_matrix(
      matrix.data());
  return eigen_matrix.block<3, 1>(0, 3).cast<double>();
}

// transforms the rotation matrix in quaternion values to be more easily
// smoothed
Eigen::Quaterniond ORB_SLAM3::ObjectPoseEMAFilter::rotationMatrixToQuaternion(
    const Eigen::Matrix3d &R) {
  return Eigen::Quaterniond(R);
}
#include <iostream>
Eigen::Vector3d
ORB_SLAM3::ObjectPoseEMAFilter::getEulerAngles(const Eigen::Matrix3d &mr) {
  // Convert rotation matrix to Euler angles (ZYX order)
  return mr.eulerAngles(2, 1, 0); // yaw, pitch, roll
}

// combines rotation, scaling, and translation into a pose matrix
Eigen::Matrix4f ORB_SLAM3::ObjectPoseEMAFilter::constructPoseMatrix(
    const Eigen::Vector3d &translationValues,
    const Eigen::Quaterniond &quaternionValues,
    const Eigen::Matrix3d &scalingMatrix) {

  Eigen::Matrix4d composeMatrixDouble = Eigen::Matrix4d::Identity();

  // Converts the rotation quaternion to a rotation matrix
  Eigen::Matrix3d rotationMatrix =
      quaternionValues.normalized().toRotationMatrix();

  // Multiplies the rotation matix by the scaling matrix
  Eigen::Matrix3d rotationAndScaling = rotationMatrix * scalingMatrix;

  // fills the pose matrix witht the rotation matrix and translation vector to
  // be return later
  composeMatrixDouble.block<3, 3>(0, 0) = rotationAndScaling;
  composeMatrixDouble.block<3, 1>(0, 3) = translationValues;

  // Casts to float for the pose to be correct
  return composeMatrixDouble.cast<float>();
}

// Checks if the change in dt is significant
bool ORB_SLAM3::ObjectPoseEMAFilter::isTheChangeInDtSignificant(double dt) {
  if (mPreviousDtValue == 0.0) {
    return true; // Always significant for the very first dt
  }
  // Uses the configured threshold to determine
  // if the change was significant
  return dt > mPreviousDtValue * mDtChangeSignificanceFactor ||
         dt < mPreviousDtValue / mDtChangeSignificanceFactor;
}

Eigen::Matrix3d ObjectPoseEMAFilter::extractRotationWithoutScale(
    const Eigen::Matrix4d &poseMatrix, const double scale) {

  Eigen::Matrix3d rotationAndScaling = poseMatrix.block<3, 3>(0, 0).cast<double>();

  Eigen::Matrix3d rotationMatrix;
  if (std::abs(scale) < 1e-9) { // Check for near-zero scale
    // Handle singular case, perhaps return identity or an error
    rotationMatrix = rotationAndScaling; // Fallback
  } else {
    rotationMatrix = rotationAndScaling / scale;
  }

  return rotationMatrix;
}

Eigen::Matrix4f ORB_SLAM3::ObjectPoseEMAFilter::filterPose(
    const Eigen::Matrix4f &currentRawPose, float scale) {

  auto startTime = std::chrono::steady_clock::now();
  // Extract translation, rotation, and scaling from the input pose
  Eigen::Vector3d translation = getTranslation(currentRawPose);
  Eigen::Matrix3d rotation =
      extractRotationWithoutScale(currentRawPose.cast<double>(), (double)scale);
  Eigen::Quaterniond quaternion = rotationMatrixToQuaternion(rotation);

  if (mIsFirstPose) {
    // if its the first pose no smoothing is applied
    mCurrentSmoothedPose.translation = translation;
    mCurrentSmoothedPose.rotation = quaternion;
    mCurrentSmoothedPose.lastUpdateTimestamp =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count();
    mPreviousSmoothedPoseForSecondPass = mCurrentSmoothedPose;
    mIsFirstPose = false;

  } else {
    // Calculate delta t (dt) to later calculate the smoothing factor alpha
    double current_time =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count();
    double dt = current_time - mCurrentSmoothedPose.lastUpdateTimestamp;
    mCurrentSmoothedPose.lastUpdateTimestamp = current_time;

    // checks if the change in dt is significant enough to change the smoothing
    // factor
    if (isTheChangeInDtSignificant(dt)) {
      mCurrentSmoothingFactorAlpha =
          1.0 - std::exp(-dt / mSmoothingTimeConstant);
      mPreviousDtValue = dt;
    }
    // first pass
    // applies y[n] = (1 - alpha) * x[n] + alpha * y[n-1]
    mCurrentSmoothedPose.translation =
        mCurrentSmoothedPose.translation +
        mCurrentSmoothingFactorAlpha *
            (translation - mCurrentSmoothedPose.translation);

    // uses Spherical linear interpolation (SLERP) for smoothing rotation
    if (mCurrentSmoothedPose.rotation.dot(quaternion) < 0) {
      quaternion = Eigen::Quaterniond(-quaternion.w(), -quaternion.x(),
                                      -quaternion.y(), -quaternion.z());
    }
    mCurrentSmoothedPose.rotation = mCurrentSmoothedPose.rotation.slerp(
        mCurrentSmoothingFactorAlpha, quaternion);
    mCurrentSmoothedPose.rotation.normalize();

    Eigen::Vector3d translation1Input = mCurrentSmoothedPose.translation;
    Eigen::Quaterniond quaternion1Input = mCurrentSmoothedPose.rotation;

    // 2nd pass
    mPreviousSmoothedPoseForSecondPass.lastUpdateTimestamp = current_time;
    // Apply exponential smoothing to translation for second pass
    mPreviousSmoothedPoseForSecondPass.translation =
        mPreviousSmoothedPoseForSecondPass.translation +
        mCurrentSmoothingFactorAlpha *
            (translation1Input -
             mPreviousSmoothedPoseForSecondPass.translation);

    if (mPreviousSmoothedPoseForSecondPass.rotation.dot(quaternion1Input) < 0) {
      quaternion1Input.coeffs() *= -1.0;
    }
    mPreviousSmoothedPoseForSecondPass.rotation =
        mPreviousSmoothedPoseForSecondPass.rotation.slerp(
            mCurrentSmoothingFactorAlpha, quaternion1Input);
  }

  auto endTime = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed_seconds = endTime - startTime;
  // Displays the time taken to filter to assess performance
  std::cout << "Filtering took: " << elapsed_seconds.count() << " seconds"
            << std::endl;

  Eigen::Matrix3d finalScalingMatrix = scale * Eigen::Matrix3d::Identity();

  // Returns the smoothed pose
  return constructPoseMatrix(mPreviousSmoothedPoseForSecondPass.translation,
                             mPreviousSmoothedPoseForSecondPass.rotation,
                             finalScalingMatrix);
}

} // namespace ORB_SLAM3
