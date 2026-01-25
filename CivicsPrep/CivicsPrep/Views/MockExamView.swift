import SwiftUI

struct MockExamView: View {
    @EnvironmentObject private var localization: LocalizationManager
    @EnvironmentObject private var purchaseManager: PurchaseManager
    @EnvironmentObject private var mockExamViewModel: MockExamViewModel

    var body: some View {
        NavigationStack {
            Group {
                if purchaseManager.hasMockExam {
                    if mockExamViewModel.sessionQuestions.isEmpty {
                        ExamSetupView()
                    } else if mockExamViewModel.isCompleted {
                        ExamResultsView()
                    } else {
                        ExamQuestionView()
                    }
                } else {
                    LockedMockExamView()
                }
            }
            .navigationTitle(localization.localized("tab_mock_exam"))
            .onAppear {
                if mockExamViewModel.sessionQuestions.isEmpty {
                    mockExamViewModel.reset()
                }
            }
        }
    }
}

struct LockedMockExamView: View {
    @EnvironmentObject private var localization: LocalizationManager
    @EnvironmentObject private var purchaseManager: PurchaseManager

    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "lock.fill")
                .font(.system(size: 48))
                .foregroundStyle(.secondary)
            Text(localization.localized("mock_exam_locked_title"))
                .font(.title2.bold())
            Text(localization.localized("mock_exam_locked_body"))
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
                .padding(.horizontal)
            if let product = purchaseManager.products.first(where: { $0.id == PurchaseManager.mockExamID }) {
                Button {
                    Task { _ = await purchaseManager.purchase(product) }
                } label: {
                    Text(localization.localized("purchase_unlock"))
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .padding(.horizontal)
            }
        }
        .padding()
    }
}
