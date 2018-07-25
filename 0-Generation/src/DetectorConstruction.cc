// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************

// User Classes
#include "DetectorConstruction.hh"

// G4 Classes
#include "G4Material.hh"
#include "G4NistManager.hh"

#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4PVReplica.hh"
#include "G4GlobalMagFieldMessenger.hh"
#include "G4AutoDelete.hh"

#include "G4GeometryManager.hh"
#include "G4PhysicalVolumeStore.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4SolidStore.hh"

#include "G4VisAttributes.hh"
#include "G4Colour.hh"

#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"



G4ThreadLocal G4GlobalMagFieldMessenger* DetectorConstruction::kMagFieldMessenger_ = 0; 


DetectorConstruction::DetectorConstruction() : G4VUserDetectorConstruction() {
}


DetectorConstruction::~DetectorConstruction() { 
}


G4VPhysicalVolume* DetectorConstruction::Construct() {
  // Define materials 
  DefineMaterials();
  
  // Define volumes
  return DefineVolumes();
}


void DetectorConstruction::DefineMaterials()
{ 
  G4NistManager* nistManager = G4NistManager::Instance();
  nistManager->FindOrBuildMaterial("G4_Pb");
  nistManager->FindOrBuildMaterial("G4_PbWO4");

  // Vacuum
  new G4Material("Galactic",             // name
                 1.0,                     // z
                 1.01*g/mole,            // a
                 universe_mean_density,  // density
                 kStateGas,              // state
                 2.73*kelvin,            // temp
                 3.e-18*pascal);         // pressure


 
  G4cout << *(G4Material::GetMaterialTable()) << G4endl;
}


G4VPhysicalVolume* DetectorConstruction::DefineVolumes() {
  // Geometry parameters
  // World
  const G4double kWorldXSide = 1000.0 * cm; 
  const G4double kWorldYSide = 1000.0 * cm; 
  const G4double kWorldZSide = 1000.0 * cm; 

  // Calorimeter
  // Simplified `CMS-like` PbWO4 crystal calorimeter  
  const G4int kNumOfCrystalsPerRow = 9;
  const G4int kNumOfCrystalsPerColumn = 9;

  const G4double kCrystalWidth = 2.2*cm;
  const G4double kCrystalLength = 22*cm;

  const G4double kCaloXSide = (kCrystalWidth * kNumOfCrystalsPerRow) + 1*cm;
  const G4double kCaloYSide = (kCrystalWidth * kNumOfCrystalsPerColumn) + 1*cm;
  const G4double kCaloZSide = kCrystalLength;

  /// Get materials
  G4Material* defaultMaterial = G4Material::GetMaterial("Galactic");
  G4Material* crystalMaterial = G4Material::GetMaterial("G4_PbWO4");


  const G4bool kCheckOverlaps = true;

 
  if ( not defaultMaterial or not crystalMaterial ) {
    G4ExceptionDescription msg;
    msg << "Cannot retrieve materials already defined."; 
    G4Exception("DetectorConstruction::DefineVolumes()",
                "MyCode0001",   //
                FatalException, //
                msg);
  }  

  ///////////////////////////////////
  // World
  //////////////////////////////////////
  G4VSolid* world_solid = new G4Box(
      "World",           // its name
      kWorldXSide / 2,
      kWorldYSide / 2,
      kWorldZSide / 2); // its size
                         
  G4LogicalVolume* world_logic = new G4LogicalVolume(
      world_solid,      // its solid
      defaultMaterial,  // its material
      "World");         // its name
                                   
  G4VPhysicalVolume* world_phys = new G4PVPlacement(
      0,                // no rotation
      G4ThreeVector(),  // at (0,0,0)
      world_logic,      // its logical volume                         
      "World",          // its name
      0,                // its mother  volume
      false,            // no boolean operation
      0,                // copy number
      kCheckOverlaps);  // checking overlaps 
 

  ///////////////////////////////////////////////////
  // Calorimeter segments
  // Simplified 'CMS-like' PbWO4' crystal calorimeter
  ///////////////////////////////////////////////////
  G4VSolid* calo_solid = new G4Box(
      "CMS_ECAL",       // its name
      kCaloXSide / 2.0, // size
      kCaloYSide / 2.0,
      kCaloZSide / 2.0);
                         
  G4LogicalVolume* calo_logic = new G4LogicalVolume(
      calo_solid,     // its solid
      defaultMaterial,  // its material
      "CMS_ECAL");

  G4double xpos = 0.0;
  G4double ypos = 0.0;
  G4double zpos = 100.0 * cm;
  new G4PVPlacement(
      0,                   // no rotation
      G4ThreeVector(xpos, ypos, zpos),
      calo_logic,          // its logical volume                         
      "CMS_ECAL",          // its name
      world_logic,         // its mother  volume
      false,               // no boolean operation
      1,                   // copy number
      kCheckOverlaps);     // checking overlaps 
 
  ///////////////////////////////
  // Crystals
  ////////////////////////////////
  G4VSolid* crystal_solid = new G4Box(
      "CrystalS",            // const G4String &pName: its name
      kCrystalWidth / 2,     // G4double pX
      kCrystalWidth / 2,     // G4double pY
      kCrystalLength / 2);   // G4double pZ

  crystal_logic_ = new G4LogicalVolume(
      crystal_solid,      // its solid
      crystalMaterial,    // its material
      "CrystalLV");       // its name
 
  for(G4int row = 0; row < kNumOfCrystalsPerRow; row++) {
    for(G4int col = 0; col < kNumOfCrystalsPerColumn; col++) {

      G4int idx = row * kNumOfCrystalsPerRow + col;

      G4cout << "make " << idx << "th crystal." << G4endl;

      G4double crystal_x = (row * kCrystalWidth) - (kCrystalWidth * kNumOfCrystalsPerRow / 2);
      G4double crystal_y = (col * kCrystalWidth) - (kCrystalWidth * kNumOfCrystalsPerColumn / 2);
      G4ThreeVector crystal_pos(crystal_x, crystal_y, 0);

      crystal_phys_[idx] = new G4PVPlacement(
          0,                // pRot: no rotation
          crystal_pos,      // tlate: translation
          crystal_logic_,   // pCurrentLogical
          "CrystalPV",      // pName: its name
          calo_logic,       // pMotherLogical
          false,            // pMany
          idx);             // pCopyNo
    }
  }

  // Visualization attributes
  world_logic->SetVisAttributes(G4VisAttributes::Invisible);

  G4VisAttributes* calo_vis_att = new G4VisAttributes(G4Colour(1.0, 1.0, 1.0));
  G4VisAttributes* crystal_vis_att = new G4VisAttributes(G4Colour(1.0, 1.0, 0.0));

  calo_logic->SetVisAttributes(calo_vis_att);
  crystal_logic_->SetVisAttributes(crystal_vis_att);

  // Always return the physical World
  return world_phys;
}


void DetectorConstruction::ConstructSDandField()
{ 
  // Create global magnetic field messenger.
  // Uniform magnetic field is then created automatically if
  // the field value is not zero.
  G4ThreeVector fieldValue = G4ThreeVector();
  kMagFieldMessenger_ = new G4GlobalMagFieldMessenger(fieldValue);
  kMagFieldMessenger_->SetVerboseLevel(1);
  
  // Register the field messenger for deleting
  G4AutoDelete::Register(kMagFieldMessenger_);
}
