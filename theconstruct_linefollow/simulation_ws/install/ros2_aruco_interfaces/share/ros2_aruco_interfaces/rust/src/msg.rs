pub mod rmw {
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[link(name = "ros2_aruco_interfaces__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__ros2_aruco_interfaces__msg__ArucoMarkers() -> *const std::os::raw::c_void;
}

#[link(name = "ros2_aruco_interfaces__rosidl_generator_c")]
extern "C" {
    fn ros2_aruco_interfaces__msg__ArucoMarkers__init(msg: *mut ArucoMarkers) -> bool;
    fn ros2_aruco_interfaces__msg__ArucoMarkers__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<ArucoMarkers>, size: usize) -> bool;
    fn ros2_aruco_interfaces__msg__ArucoMarkers__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<ArucoMarkers>);
    fn ros2_aruco_interfaces__msg__ArucoMarkers__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<ArucoMarkers>, out_seq: *mut rosidl_runtime_rs::Sequence<ArucoMarkers>) -> bool;
}

// Corresponds to ros2_aruco_interfaces__msg__ArucoMarkers
#[repr(C)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct ArucoMarkers {
    pub header: std_msgs::msg::rmw::Header,
    pub marker_ids: rosidl_runtime_rs::Sequence<i64>,
    pub poses: rosidl_runtime_rs::Sequence<geometry_msgs::msg::rmw::Pose>,
}



impl Default for ArucoMarkers {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !ros2_aruco_interfaces__msg__ArucoMarkers__init(&mut msg as *mut _) {
        panic!("Call to ros2_aruco_interfaces__msg__ArucoMarkers__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for ArucoMarkers {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { ros2_aruco_interfaces__msg__ArucoMarkers__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { ros2_aruco_interfaces__msg__ArucoMarkers__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { ros2_aruco_interfaces__msg__ArucoMarkers__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for ArucoMarkers {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for ArucoMarkers where Self: Sized {
  const TYPE_NAME: &'static str = "ros2_aruco_interfaces/msg/ArucoMarkers";
  fn get_type_support() -> *const std::os::raw::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__ros2_aruco_interfaces__msg__ArucoMarkers() }
  }
}


}  // mod rmw


#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct ArucoMarkers {
    pub header: std_msgs::msg::Header,
    pub marker_ids: Vec<i64>,
    pub poses: Vec<geometry_msgs::msg::Pose>,
}



impl Default for ArucoMarkers {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(crate::msg::rmw::ArucoMarkers::default())
  }
}

impl rosidl_runtime_rs::Message for ArucoMarkers {
  type RmwMsg = crate::msg::rmw::ArucoMarkers;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Owned(msg.header)).into_owned(),
        marker_ids: msg.marker_ids.into(),
        poses: msg.poses
          .into_iter()
          .map(|elem| geometry_msgs::msg::Pose::into_rmw_message(std::borrow::Cow::Owned(elem)).into_owned())
          .collect(),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(&msg.header)).into_owned(),
        marker_ids: msg.marker_ids.as_slice().into(),
        poses: msg.poses
          .iter()
          .map(|elem| geometry_msgs::msg::Pose::into_rmw_message(std::borrow::Cow::Borrowed(elem)).into_owned())
          .collect(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      header: std_msgs::msg::Header::from_rmw_message(msg.header),
      marker_ids: msg.marker_ids
          .into_iter()
          .collect(),
      poses: msg.poses
          .into_iter()
          .map(geometry_msgs::msg::Pose::from_rmw_message)
          .collect(),
    }
  }
}


