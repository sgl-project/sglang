# NaturaLingo (iOS MVP)

## Overview
NaturaLingo is a SwiftUI iOS 17+ MVP for preparing the U.S. Citizenship test with bilingual content in Simplified Chinese and Spanish. Study mode is free with ads, and paid unlocks remove ads and enable mock exams.

## Requirements
- Xcode 15+
- iOS 17+ simulator/device

## Setup
1. Open `CivicsPrep/CivicsPrep.xcodeproj` in Xcode.
2. Select an iOS 17+ simulator and run.

## AdMob integration (placeholder)
This MVP includes a banner placeholder view that can be replaced with a real AdMob banner.
1. Add GoogleMobileAds via Swift Package Manager.
2. Replace `BannerAdPlaceholder` in `AdsManager.swift` with a `GADBannerView` wrapper.
3. Update the Ad Unit IDs in the AdsManager to match your AdMob account.

## In-App Purchases (StoreKit 2)
- Remove Ads (non-consumable): `com.example.civicsprep.removeads`
- Mock Exam (non-consumable): `com.example.civicsprep.mockexam`

### App Store Connect
1. Create the two non-consumable products above.
2. Match the product IDs in `PurchaseManager.swift`.

### Testing StoreKit in Xcode
1. In Xcode, select **Product > Scheme > Edit Scheme**.
2. Under **Run > Options**, set the StoreKit configuration file to `MVP.storekit`.
3. Run the app and use the StoreKit test sheet to simulate purchases.

## Local content
Questions live in `CivicsPrep/CivicsPrep/Resources/questions.json`. The data model supports adding English later.
