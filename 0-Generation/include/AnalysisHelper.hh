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

#ifndef ANALYSISHELPER_HH_
#define ANALYSISHELPER_HH_

//
#include "globals.hh"

// ROOT Classes
#include "TString.h"

// G4 Classes Forward
class G4Run;
class G4Event;
class G4Step;

// ROOT Classes Forward
class TFile;
class TTree;

// Constants
// TODO find better way
const static  G4int kNumCrystals = 81;



class AnalysisHelper {
 public:
  AnalysisHelper();
  ~AnalysisHelper();
  static AnalysisHelper* GetInstance();

  void DoBeginOfRunAction(const G4Run* run);
  void DoEndOfRunAction(const G4Run* run);
  void DoBeginOfEventAction(const G4Event* event);
  void DoEndOfEventAction(const G4Event* event);
  void DoSteppingAction(const G4Step* step);

  // DoBeginOfRunAction
  void MakeBranch();

  // DoEndOfRunAction
  void PrintRunSummary();

  // DoBeginOfEventAction
  void ResetBranch();

  // DoEndOfEventAction


  // DoSteppingAction


 private:
  static AnalysisHelper* singleton;

  TString out_path_;
  TString tree_name_;
  TFile* root_file_;
  TTree* tree_;

  // Branches
  G4double energy_deposit_[kNumCrystals];
  G4double total_energy_;


};


   

#endif // ANALYSISHELPER_HH_
