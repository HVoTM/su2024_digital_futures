// Include necessary plugins and libraries for performance
#include <ignition/math/Pose3.hh>
#include "gazebo/physics/physics.hh"
#include "gazebo/common/common.hh"
#include "gazebo/gazebo.hh"

// Creates a world plugin for use inheriting from the WorldPlugin 
namespace gazebo
{
class Factory : public WorldPlugin
{
  // Load() method uses 3 different methods for model insertion
  public: void Load(physics::WorldPtr _parent, sdf::ElementPtr /*_sdf*/)
  {
    // Option 1: Insert model from file in the resource path via function call.
    // The filename must be in the GAZEBO_MODEL_PATH environment variable.
    _parent->InsertModelFile("model://box");

    // Option 2: Insert model from string via function call.
    // Insert a sphere model from string
    sdf::SDF sphereSDF;
    sphereSDF.SetFromString(
       "<sdf version ='1.4'>\
          <model name ='sphere'>\
            <pose>1 0 0 0 0 0</pose>\
            <link name ='link'>\
              <pose>0 0 .5 0 0 0</pose>\
              <collision name ='collision'>\
                <geometry>\
                  <sphere><radius>0.5</radius></sphere>\
                </geometry>\
              </collision>\
              <visual name ='visual'>\
                <geometry>\
                  <sphere><radius>0.5</radius></sphere>\
                </geometry>\
              </visual>\
            </link>\
          </model>\
        </sdf>");
    // Demonstrate using a custom model name.
    sdf::ElementPtr model = sphereSDF.Root()->GetElement("model");
    model->GetAttribute("name")->SetFromString("unique_sphere");
    _parent->InsertModelSDF(sphereSDF);

    // Option 3: Insert model from file via message passing mechanism
    // most useful for standalone applications that communicate with Gazebo
    // over a network connection
    {
      // Create a new transport node
      transport::NodePtr node(new transport::Node());

      // Initialize the node with the world name
      node->Init(_parent->Name());

      // Create a publisher on the ~/factory topic
      transport::PublisherPtr factoryPub =
      node->Advertise<msgs::Factory>("~/factory");

      // Create the message
      msgs::Factory msg;

      // Model file to load
      msg.set_sdf_filename("model://cylinder");

      // Pose to initialize the model to
      msgs::Set(msg.mutable_pose(),
          ignition::math::Pose3d(
            ignition::math::Vector3d(1, -2, 0),
            ignition::math::Quaterniond(0, 0, 0)));

      // Send the message
      factoryPub->Publish(msg);
    }
  }
};

// Register this plugin with the simulator
GZ_REGISTER_WORLD_PLUGIN(Factory)
}