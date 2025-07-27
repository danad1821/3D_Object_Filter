
#include <eigen3/Eigen/Dense>
#include <memory>
#include <mutex>

#include "ObjectFilterManager.h"
using namespace std;

namespace ORB_SLAM3 {
ObjectFilterManager::ObjectFilterManager(const std::string &strSettingPath,
                                         bool filterActive)
    : mStrSettingPath(strSettingPath), mFilterActive(filterActive) {
  setSettingStr(strSettingPath);
}

ObjectPoseEMAFilter &ObjectFilterManager::getOrCreateFilter(int objId) {
  if (!mFilterActive) {
    throw std::runtime_error("Object filtering is not active.");
  }
  unique_lock<mutex> lock(mMutexFilters);
  auto it = mObjectFilters.find(objId);
  if (it != mObjectFilters.end()) {
    return *(it->second);
  } else {
    auto filter = std::make_unique<ObjectPoseEMAFilter>(mStrSettingPath);
    ObjectPoseEMAFilter *filterPtr = filter.get();
    mObjectFilters[objId] = std::move(filter);
    return *filterPtr;
  }
}

// given an object ID and a raw pose, filter the pose using the corresponding filter
// if the filter does not exist, it will be created
// if filtering is not active, it will return the raw pose without filtering
Eigen::Matrix4f
ObjectFilterManager::filterObjectPose(int objId, const Eigen::Matrix4f &rawPose,
                                      float scale) {
  if (!mFilterActive) {
    return rawPose;
  }
  ObjectPoseEMAFilter &filter = getOrCreateFilter(objId);
  return filter.filterPose(rawPose, scale);
}

void ObjectFilterManager::removeFilter(int objId) {
  mObjectFilters.erase(objId);
}

bool ObjectFilterManager::hasFilter(int objId) const {
  return mObjectFilters.find(objId) != mObjectFilters.end();
}

void ObjectFilterManager::setSettingStr(const std::string &strSettingPath) {
  mStrSettingPath = strSettingPath;
}

std::string ObjectFilterManager::getSettingStr() { return mStrSettingPath; }

} // namespace ORB_SLAM3